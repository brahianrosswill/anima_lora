"""Dataset plumbing for the Anima tagger (dual-encoder, hard-routed).

* :class:`TaggerManifest` — loads ``dataset.json`` (the per-stem
  image-path + multi-hot tag indices + rating-class index + people-count
  class index emitted by ``python -m scripts.anima_tagger.cli --mode
  build_vocab``).
* :class:`FeatureCacheBuilder` — mean-pool cache (``pool_kind="mean"``).
  Encodes each manifest image through a frozen PE trunk, mean-pools over
  patch tokens, writes a per-stem ``[d_enc] fp32`` safetensors. Idempotent.
* :class:`TokenCacheBuilder` — full token cache (``pool_kind="map"``).
  Encodes each manifest image through a frozen PE trunk, writes a per-stem
  ``[T, d_enc] bf16`` safetensors (CLS at row 0). Storage is ~1.2 MB /
  stem at PE-Core-L14-336; the win is that swapping pool architectures no
  longer requires re-encoding.
* :class:`CachedDualDataset` + :func:`collate_dual_token_batch` — the lazy
  bucket-grouped dataset feeding both encoders. Each side independently
  loads a pooled ``[d_enc]`` feature (mean) or ``[T, d_enc]`` tokens (map);
  T is constant within a bucket so the collate just stacks per side.
* :class:`BucketBatchSampler` — groups dataset indices into
  shape-homogeneous batches per (main, aux) bucket pair.

Cache layout is locked into each builder's file format — swap pooling /
storage layout → invalidate the cache dir → rebuild.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file as st_load
from safetensors.torch import save_file as st_save
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

from library.datasets.image_utils import IMAGE_TRANSFORMS
from library.vision.buckets import (
    BucketSpec,
    bucket_pixel_size,
    pick_bucket,
)
from library.vision.encoder import (
    VisionEncoderBundle,
    encode_pe_from_imageminus1to1,
    load_pe_encoder,
)

logger = logging.getLogger(__name__)


def pil_resize_to_bucket(img: Image.Image, spec: BucketSpec) -> Image.Image:
    """LANCZOS-resize a PIL image to its closest bucket size for ``spec``.

    Pre-resizing on the PIL side (high quality LANCZOS) avoids decoding
    multi-megapixel source images into a tensor only to bilinear-resize
    them down inside the encoder. Speeds up cache builds 5–10× on
    high-resolution corpora and removes a quality penalty (LANCZOS >
    bilinear for severe downscales).
    """
    w, h = img.size
    h_p, w_p = pick_bucket(h, w, spec)
    target_h, target_w = bucket_pixel_size((h_p, w_p), spec)
    if (h, w) != (target_h, target_w):
        img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    return img


class _ResizeDataset(Dataset):
    """CPU-side decode + bucket-resize + IMAGE_TRANSFORMS for one stem.

    Returns ``(stem, tensor[C,H,W] | None, err)``. Errors are surfaced as
    a non-empty string so the consumer can log and continue.
    """

    def __init__(
        self,
        stems: Sequence[str],
        image_paths: Sequence[Path],
        spec: BucketSpec,
    ):
        self._stems = list(stems)
        self._paths = list(image_paths)
        self._spec = spec

    def __len__(self) -> int:
        return len(self._stems)

    def __getitem__(self, k: int):
        stem = self._stems[k]
        path = self._paths[k]
        try:
            with Image.open(path) as im:
                im = pil_resize_to_bucket(im.convert("RGB"), self._spec)
                arr = np.array(im)
            tensor = IMAGE_TRANSFORMS(arr)  # [C, H, W]
            return stem, tensor, ""
        except Exception as e:
            return stem, None, f"{type(e).__name__}: {e}"


def _bucket_collate(batch):
    """Stack a shape-homogeneous batch, filtering decode failures.

    The :func:`_bucket_batches` sampler guarantees every item in a batch
    shares an aspect bucket, so the surviving tensors stack cleanly into
    ``[B, C, H, W]``. Returns ``(stems_ok, stacked | None, errs)`` where
    ``errs`` carries ``(stem, msg)`` for items that failed to decode.
    """
    stems_ok: list[str] = []
    tensors: list[torch.Tensor] = []
    errs: list[tuple[str, str]] = []
    for stem, tensor, err in batch:
        if tensor is None:
            errs.append((stem, err))
        else:
            stems_ok.append(stem)
            tensors.append(tensor)
    stacked = torch.stack(tensors, dim=0) if tensors else None
    return stems_ok, stacked, errs


def _bucket_batches(
    image_paths: Sequence[Path], spec: BucketSpec, batch_size: int
) -> tuple[list[list[int]], list[tuple[int, str]]]:
    """Group image indices by aspect bucket, then chunk into batches.

    Reads only each image's header (``PIL.Image.size`` — no pixel decode) to
    pick its bucket, so images that resize to the same pixel grid land in the
    same batch and can share one encoder forward. Returns the batch list (each
    a list of dataset indices) plus ``(index, msg)`` for unreadable headers,
    which the caller logs and skips.
    """
    by_bucket: Dict[tuple, List[int]] = defaultdict(list)
    header_errs: list[tuple[int, str]] = []
    for i, path in enumerate(image_paths):
        try:
            with Image.open(path) as im:
                w, h = im.size
            key = pick_bucket(h, w, spec)
        except Exception as e:
            header_errs.append((i, f"{type(e).__name__}: {e}"))
            continue
        by_bucket[key].append(i)

    batches: list[list[int]] = []
    for idxs in by_bucket.values():
        for j in range(0, len(idxs), batch_size):
            batches.append(idxs[j : j + batch_size])
    return batches, header_errs


def _run_pe_cache(
    *,
    stems: Sequence[str],
    image_paths: Sequence[Path],
    spec: BucketSpec,
    bundle: VisionEncoderBundle,
    num_workers: int,
    batch_size: int,
    save_one,
    desc: str,
) -> int:
    """Bucket-batched encode loop shared by the feature / token builders.

    Groups the (already-filtered-to-missing) ``stems`` by aspect bucket, runs
    one batched ``encode_pe_from_imageminus1to1(..., same_bucket=True)`` forward
    per batch, and hands each ``[T, d_enc]`` result to ``save_one(stem, feats)``.
    Per-image decode / save failures are logged and skipped. Returns the count
    of entries written.
    """
    batches, header_errs = _bucket_batches(image_paths, spec, batch_size)
    for i, err in header_errs:
        logger.warning("failed to read %s: %s", stems[i], err)

    ds = _ResizeDataset(stems=stems, image_paths=image_paths, spec=spec)
    loader = DataLoader(
        ds,
        batch_sampler=batches,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=_bucket_collate,
        pin_memory=False,
        persistent_workers=(num_workers > 0 and len(batches) > 1),
    )

    n_done = 0
    for stems_ok, img_batch, errs in tqdm(
        loader, desc=desc, unit="batch", total=len(batches)
    ):
        for stem, err in errs:
            logger.warning("failed to decode %s: %s", stem, err)
        if img_batch is None:
            continue
        try:
            feats_list = encode_pe_from_imageminus1to1(
                bundle, img_batch, same_bucket=True
            )
        except Exception as e:
            logger.warning("failed to encode batch of %d: %s", len(stems_ok), e)
            continue
        for stem, feats in zip(stems_ok, feats_list):
            try:
                save_one(stem, feats)
                n_done += 1
            except Exception as e:
                logger.warning("failed to save %s: %s", stem, e)
    return n_done


@dataclass
class TaggerManifest:
    """Trainable-sample manifest emitted by ``--mode build_vocab``."""

    stems: List[str]
    image_paths: List[Path]
    tag_indices: List[List[int]]
    rating_indices: List[int]
    people_count_indices: List[int]
    train_stems: List[str]
    val_stems: List[str]
    n_tags: int
    n_ratings: int
    n_people_counts: int

    @classmethod
    def from_path(cls, path: Path) -> "TaggerManifest":
        with open(path) as f:
            d = json.load(f)
        # ``people_count_indices`` / ``n_people_counts`` were added late; old
        # manifests rebuild on next ``build_vocab``. Until then, default to a
        # zero-length head so the trainer can detect "no people supervision"
        # cleanly (rather than crashing with a KeyError).
        people_idx = list(d.get("people_count_indices") or [])
        n_people = int(d.get("n_people_counts", 0))
        if people_idx and not n_people:
            n_people = max(people_idx) + 1
        return cls(
            stems=list(d["stems"]),
            image_paths=[Path(p) for p in d["image_paths"]],
            tag_indices=[list(idxs) for idxs in d["tag_indices"]],
            rating_indices=list(d["rating_indices"]),
            people_count_indices=people_idx,
            train_stems=list(d["split"]["train"]),
            val_stems=list(d["split"]["val"]),
            n_tags=int(d["n_tags"]),
            n_ratings=int(d["n_ratings"]),
            n_people_counts=n_people,
        )

    def stem_index(self) -> Dict[str, int]:
        return {s: i for i, s in enumerate(self.stems)}


# ── Feature cache ─────────────────────────────────────────────────────────


def _cache_path(cache_dir: Path, stem: str) -> Path:
    return cache_dir / f"{stem}.safetensors"


class FeatureCacheBuilder:
    """Build per-stem mean-pooled PE-Core features into ``cache_dir``.

    Groups missing stems by aspect bucket and runs one batched encoder
    forward per ``batch_size`` shape-homogeneous images (``_run_pe_cache``),
    keeping the GPU fed instead of one image per forward. Idempotent.
    """

    def __init__(
        self,
        manifest: TaggerManifest,
        cache_dir: Path,
        device: torch.device,
        encoder_name: str = "pe",
        dtype: torch.dtype = torch.bfloat16,
        num_workers: int = 4,
        batch_size: int = 8,
    ):
        self.manifest = manifest
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.encoder_name = encoder_name
        self.dtype = dtype
        self.num_workers = num_workers
        self.batch_size = batch_size
        self._bundle: Optional[VisionEncoderBundle] = None

    def _bundle_lazy(self) -> VisionEncoderBundle:
        if self._bundle is None:
            self._bundle = load_pe_encoder(
                self.device, name=self.encoder_name, dtype=self.dtype
            )
        return self._bundle

    def missing_stems(self) -> List[int]:
        return [
            i
            for i, stem in enumerate(self.manifest.stems)
            if not _cache_path(self.cache_dir, stem).exists()
        ]

    @torch.no_grad()
    def build(self) -> int:
        """Encode + cache every stem missing from ``cache_dir``.

        Returns the count of newly cached entries (0 if everything was
        already cached). Errors on individual images are logged and the
        loop continues — a single corrupt image shouldn't tank the run.
        """
        missing = self.missing_stems()
        if not missing:
            logger.info(
                "feature cache: all %d entries present", len(self.manifest.stems)
            )
            return 0

        logger.info(
            "feature cache: encoding %d missing entries (out of %d total)",
            len(missing),
            len(self.manifest.stems),
        )
        bundle = self._bundle_lazy()
        spec = bundle.bucket_spec
        d_enc = bundle.d_enc

        def save_one(stem: str, feats: torch.Tensor) -> None:
            pooled = feats.mean(dim=0).to(torch.float32).cpu()  # [d_enc]
            assert pooled.shape == (d_enc,), pooled.shape
            st_save({"feature": pooled}, str(_cache_path(self.cache_dir, stem)))

        n_done = _run_pe_cache(
            stems=[self.manifest.stems[i] for i in missing],
            image_paths=[self.manifest.image_paths[i] for i in missing],
            spec=spec,
            bundle=bundle,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            save_one=save_one,
            desc="pooled-pe",
        )
        logger.info("feature cache: wrote %d new entries", n_done)
        return n_done


# ── Token cache (for MAP-pool / pool_kind="map" training) ────────────────


def _token_cache_path(cache_dir: Path, stem: str) -> Path:
    return cache_dir / f"{stem}.safetensors"


class TokenCacheBuilder:
    """Build per-stem full-token PE-Core caches into ``cache_dir``.

    Writes each stem as ``{"tokens": bf16 [T, d_enc]}`` with the encoder's
    native CLS token at row 0 (use_cls=True). T varies per aspect-bucket
    (~576–588 for PE-Core-L14-336).

    Storage per stem ≈ ``T * d_enc * 2`` bytes; ~1.2 MB at PE-Core defaults.
    At 12K stems that's ~14 GB total — pay once, iterate on pool design
    freely. Missing stems are grouped by aspect bucket and encoded in
    batches of ``batch_size`` (``_run_pe_cache``) to keep the GPU saturated.
    """

    def __init__(
        self,
        manifest: TaggerManifest,
        cache_dir: Path,
        device: torch.device,
        encoder_name: str = "pe",
        dtype: torch.dtype = torch.bfloat16,
        num_workers: int = 4,
        batch_size: int = 8,
    ):
        self.manifest = manifest
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.encoder_name = encoder_name
        self.dtype = dtype
        self.num_workers = num_workers
        self.batch_size = batch_size
        self._bundle: Optional[VisionEncoderBundle] = None

    def _bundle_lazy(self) -> VisionEncoderBundle:
        if self._bundle is None:
            self._bundle = load_pe_encoder(
                self.device, name=self.encoder_name, dtype=self.dtype
            )
        return self._bundle

    def missing_stems(self) -> List[int]:
        return [
            i
            for i, stem in enumerate(self.manifest.stems)
            if not _token_cache_path(self.cache_dir, stem).exists()
        ]

    @torch.no_grad()
    def build(self) -> int:
        """Encode + cache every stem missing from ``cache_dir``.

        Returns the count of newly cached entries (0 if everything was
        already cached). Errors on individual images are logged and the
        loop continues — a single corrupt image shouldn't tank the run.
        """
        missing = self.missing_stems()
        if not missing:
            logger.info("token cache: all %d entries present", len(self.manifest.stems))
            return 0

        logger.info(
            "token cache: encoding %d missing entries (out of %d total)",
            len(missing),
            len(self.manifest.stems),
        )
        bundle = self._bundle_lazy()
        spec = bundle.bucket_spec
        d_enc = bundle.d_enc

        def save_one(stem: str, feats: torch.Tensor) -> None:
            assert feats.shape[-1] == d_enc, feats.shape
            # Stash as bf16 — encoder's native dtype, halves cache size
            # vs fp32 with no quality loss on the downstream pool (the
            # MAP head's LayerNorm + Linear projects back into fp32 /
            # autocast-bf16 anyway).
            tokens = feats.to(torch.bfloat16).cpu().contiguous()
            st_save({"tokens": tokens}, str(_token_cache_path(self.cache_dir, stem)))

        n_done = _run_pe_cache(
            stems=[self.manifest.stems[i] for i in missing],
            image_paths=[self.manifest.image_paths[i] for i in missing],
            spec=spec,
            bundle=bundle,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            save_one=save_one,
            desc="tokens-pe",
        )
        logger.info("token cache: wrote %d new entries", n_done)
        return n_done


# ── Dual-encoder cache (PE-Core + PE-Spatial; per-side pool kind) ────────


class CachedDualDataset(Dataset):
    """Lazy per-stem ``(feat_main, feat_aux, multi_hot, rating, people, bucket_pair)``.

    Each side independently picks ``"mean"`` or ``"map"``:
      * ``"mean"`` → load pooled cache (``{stem}.safetensors`` with key
        ``"feature"``), shape ``[D]``. Bucket key is ``None`` for that side
        (no shape variation → no batch-bucketing constraint).
      * ``"map"`` → load token cache (``{stem}.safetensors`` with key
        ``"tokens"``), shape ``[T, D]``. Bucket key is ``(h_p, w_p)`` derived
        from T via the encoder's :class:`BucketSpec`.

    The compound bucket key is ``(main_bucket | None, aux_bucket | None)``;
    :class:`BucketBatchSampler` groups by it, so within-batch shape
    homogeneity holds for whichever side(s) need it. When PE-Core uses
    mean and PE-Spatial uses map (the most common asymmetric setup),
    batches are grouped by aux bucket only.

    ``cache_dir`` / ``cache_dir_aux`` should be the per-side outputs of
    :func:`scripts.anima_tagger.caches.cache_dir_for` (i.e.
    ``.cache/pooled-<encoder>/`` for mean, ``.cache/tokens-<encoder>/`` for
    map). Spec is only used when the side is map (to map T → bucket); pass
    ``None`` for the spec on a mean side.

    Stems present in only one cache are skipped (logged); typical cause is
    an asymmetric incremental cache build.
    """

    def __init__(
        self,
        manifest: TaggerManifest,
        cache_dir: Path,
        pool_kind: str,
        spec: Optional[BucketSpec],
        cache_dir_aux: Path,
        pool_kind_aux: str,
        spec_aux: Optional[BucketSpec],
        stems_subset: Optional[Sequence[str]] = None,
    ):
        if pool_kind not in ("mean", "map"):
            raise ValueError(f"pool_kind must be 'mean' or 'map', got {pool_kind!r}")
        if pool_kind_aux not in ("mean", "map"):
            raise ValueError(
                f"pool_kind_aux must be 'mean' or 'map', got {pool_kind_aux!r}"
            )
        if pool_kind == "map" and spec is None:
            raise ValueError("pool_kind='map' requires a BucketSpec for the main side")
        if pool_kind_aux == "map" and spec_aux is None:
            raise ValueError(
                "pool_kind_aux='map' requires a BucketSpec for the aux side"
            )

        idx_of = manifest.stem_index()
        if stems_subset is None:
            stems_subset = manifest.stems

        # Per-side T → bucket map (only used on map sides). For mean sides
        # the dict is empty and the bucket key is fixed at None.
        def _bucket_map(spec: Optional[BucketSpec]) -> Dict[int, tuple[int, int]]:
            if spec is None:
                return {}
            return {h * w + (1 if spec.use_cls else 0): (h, w) for h, w in spec.buckets}

        token_to_bucket = _bucket_map(spec)
        token_to_bucket_aux = _bucket_map(spec_aux)

        kept_stems: List[str] = []
        kept_paths: List[Path] = []
        kept_paths_aux: List[Path] = []
        kept_tag_idx: List[List[int]] = []
        kept_rating_idx: List[int] = []
        kept_people_idx: List[int] = []
        kept_buckets: List[
            tuple[Optional[tuple[int, int]], Optional[tuple[int, int]]]
        ] = []
        has_people = bool(manifest.people_count_indices)
        n_missing_main = 0
        n_missing_aux = 0
        n_unknown_bucket = 0
        for stem in stems_subset:
            i = idx_of.get(stem)
            if i is None:
                n_missing_main += 1
                continue
            # Both pooled and token caches use the same per-stem filename;
            # the differentiator is the cache_dir (.cache/pooled-X/ vs
            # .cache/tokens-X/) and the safetensors key inside.
            cache_file = cache_dir / f"{stem}.safetensors"
            cache_file_aux = cache_dir_aux / f"{stem}.safetensors"
            if not cache_file.exists():
                n_missing_main += 1
                continue
            if not cache_file_aux.exists():
                n_missing_aux += 1
                continue
            bucket_main: Optional[tuple[int, int]] = None
            if pool_kind == "map":
                with safe_open(str(cache_file), framework="pt") as f:
                    T_main = int(f.get_slice("tokens").get_shape()[0])
                bucket_main = token_to_bucket.get(T_main)
                if bucket_main is None:
                    n_unknown_bucket += 1
                    continue
            bucket_aux: Optional[tuple[int, int]] = None
            if pool_kind_aux == "map":
                with safe_open(str(cache_file_aux), framework="pt") as f:
                    T_aux = int(f.get_slice("tokens").get_shape()[0])
                bucket_aux = token_to_bucket_aux.get(T_aux)
                if bucket_aux is None:
                    n_unknown_bucket += 1
                    continue
            kept_stems.append(stem)
            kept_paths.append(cache_file)
            kept_paths_aux.append(cache_file_aux)
            kept_tag_idx.append(manifest.tag_indices[i])
            kept_rating_idx.append(manifest.rating_indices[i])
            kept_people_idx.append(
                manifest.people_count_indices[i] if has_people else 0
            )
            kept_buckets.append((bucket_main, bucket_aux))
        if not kept_stems:
            raise RuntimeError(
                f"no paired sidecars in {cache_dir} + {cache_dir_aux} for the "
                f"requested stems (n_requested={len(stems_subset)}, "
                f"n_missing_main={n_missing_main}, n_missing_aux={n_missing_aux}, "
                f"n_unknown_bucket={n_unknown_bucket})"
            )
        if n_missing_main or n_missing_aux:
            logger.warning(
                "CachedDualDataset: missing main=%d aux=%d (out of %d "
                "requested) - those stems are skipped",
                n_missing_main,
                n_missing_aux,
                len(stems_subset),
            )
        if n_unknown_bucket:
            logger.warning(
                "CachedDualDataset: %d stems had unexpected token counts "
                "(not in spec.buckets) and were skipped",
                n_unknown_bucket,
            )
        self.stems = kept_stems
        self.paths = kept_paths
        self.paths_aux = kept_paths_aux
        self.buckets = kept_buckets
        self.pool_kind = pool_kind
        self.pool_kind_aux = pool_kind_aux
        self.multi_hot = torch.zeros(len(kept_stems), manifest.n_tags)
        for row, idxs in enumerate(kept_tag_idx):
            self.multi_hot[row, idxs] = 1.0
        self.rating_idx = torch.tensor(kept_rating_idx, dtype=torch.long)
        self.people_idx = torch.tensor(kept_people_idx, dtype=torch.long)
        self.n_tags = manifest.n_tags
        self.n_ratings = manifest.n_ratings
        self.n_people_counts = manifest.n_people_counts
        self.spec = spec
        self.spec_aux = spec_aux
        # Peek the first sidecar of each side to record d_in / d_in_aux.
        # Key differs by pool_kind ("feature" for mean, "tokens" for map).
        self.d_in = self._peek_d(kept_paths[0], pool_kind)
        self.d_in_aux = self._peek_d(kept_paths_aux[0], pool_kind_aux)

    @staticmethod
    def _peek_d(path: Path, kind: str) -> int:
        key = "feature" if kind == "mean" else "tokens"
        with safe_open(str(path), framework="pt") as f:
            return int(f.get_slice(key).get_shape()[-1])

    def __len__(self) -> int:
        return len(self.stems)

    @staticmethod
    def _load_one(path: Path, kind: str) -> torch.Tensor:
        key = "feature" if kind == "mean" else "tokens"
        return st_load(str(path))[key]

    def __getitem__(self, idx: int):
        feat = self._load_one(self.paths[idx], self.pool_kind)
        feat_aux = self._load_one(self.paths_aux[idx], self.pool_kind_aux)
        return (
            feat,
            feat_aux,
            self.multi_hot[idx],
            self.rating_idx[idx],
            self.people_idx[idx],
            self.buckets[idx],
        )


# Back-compat alias — earlier code (and the smoke tests) refer to the
# original name. The new class generalizes the original; callers that
# imported the old name keep working.
CachedDualTokenDataset = CachedDualDataset


def collate_dual_token_batch(batch):
    """Stack a same-bucket-pair batch into
    ``(feat_main, feat_aux, multi_hot, rating, people, bucket_pair)``.

    BucketBatchSampler guarantees both shapes are constant within a batch
    (sampler groups by the compound ``(main_bucket | None, aux_bucket | None)``
    key). torch.stack works whether each side is rank-2 (mean-pool) or
    rank-3 (token sequence).
    """
    feat = torch.stack([b[0] for b in batch], dim=0)  # [B, ...] (rank depends on side)
    feat_aux = torch.stack([b[1] for b in batch], dim=0)  # [B, ...]
    multi_hot = torch.stack([b[2] for b in batch], dim=0)  # [B, n_tags]
    rating_idx = torch.stack([b[3] for b in batch], dim=0)  # [B]
    people_idx = torch.stack([b[4] for b in batch], dim=0)  # [B]
    bucket_pair = batch[0][5]
    return feat, feat_aux, multi_hot, rating_idx, people_idx, bucket_pair


class BucketBatchSampler(Sampler[List[int]]):
    """Yields batches of indices that share a single bucket.

    Within each epoch: shuffle the per-bucket index pools, chunk into
    batches of ``batch_size`` (drop_last=False — partial trailing batches
    are kept since dataset sizes don't divide evenly), then shuffle the
    batch order across buckets so the encoder doesn't see all of one
    aspect ratio in a row.
    """

    def __init__(
        self,
        buckets: Sequence[tuple[int, int]],
        batch_size: int,
        seed: int = 42,
        shuffle: bool = True,
    ):
        self.buckets = list(buckets)
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self._epoch = 0
        # Group sample indices by bucket key.
        self._by_bucket: Dict[tuple[int, int], List[int]] = defaultdict(list)
        for i, b in enumerate(self.buckets):
            self._by_bucket[b].append(i)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        rng = torch.Generator().manual_seed(self.seed + self._epoch)
        all_batches: List[List[int]] = []
        for _, idxs in sorted(self._by_bucket.items()):
            order = idxs[:]
            if self.shuffle:
                perm = torch.randperm(len(order), generator=rng).tolist()
                order = [order[k] for k in perm]
            for s in range(0, len(order), self.batch_size):
                all_batches.append(order[s : s + self.batch_size])
        if self.shuffle:
            perm = torch.randperm(len(all_batches), generator=rng).tolist()
            all_batches = [all_batches[k] for k in perm]
        yield from all_batches

    def __len__(self) -> int:
        n = 0
        for idxs in self._by_bucket.values():
            n += (len(idxs) + self.batch_size - 1) // self.batch_size
        return n
