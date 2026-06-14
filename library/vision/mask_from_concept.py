"""SAM3 open-vocabulary text→mask — the FLAIR-edit convenience front-end.

The one genuinely new capability behind the "just tell me the change" UX in
``docs/proposal/flair_edit.md`` Phase B: map a short concept phrase (e.g.
``"eyes"``, ``"hair"``) to a binary region mask using SAM3 — the exact same
``Sam3Processor.set_image`` / ``set_text_prompt`` path ``make mask`` drives
(``scripts/preprocess/generate_masks.py``), here as a single per-edit call with
no disk caching.

This is best-effort localization: when the concept isn't found the mask is
empty and the caller should fall back to an explicit ``--flair_mask`` (the
correctness anchor). The model is loaded lazily and cached per process so a
multi-edit session pays the SAM3 load once.
"""

from __future__ import annotations

import numpy as np

from library.env import resolve_under_home

_DEFAULT_SAM3_CKPT = "models/sam3/sam3.pt"

# Process-global SAM3 processor (the load is heavy; reuse across edits).
_PROCESSOR = None
_PROCESSOR_DEVICE = None


def _get_processor(device: str, checkpoint: str | None):
    global _PROCESSOR, _PROCESSOR_DEVICE
    if _PROCESSOR is not None and _PROCESSOR_DEVICE == device:
        return _PROCESSOR

    # Upstream sam3 pins numpy<2 and uses the removed ``np.bool`` alias.
    if not hasattr(np, "bool"):
        np.bool = np.bool_  # type: ignore[attr-defined]

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    ckpt = str(resolve_under_home(checkpoint or _DEFAULT_SAM3_CKPT))
    model = build_sam3_image_model(
        device=device, eval_mode=True, checkpoint_path=ckpt, load_from_HF=False
    )
    _PROCESSOR = Sam3Processor(model)
    _PROCESSOR_DEVICE = device
    return _PROCESSOR


def mask_from_concept(
    image,
    phrase: str,
    *,
    threshold: float = 0.5,
    dilate: int = 0,
    device: str = "cuda",
    checkpoint: str | None = None,
) -> np.ndarray:
    """Segment an open-vocabulary concept → a binary ``HxW`` mask in ``{0, 1}``.

    Args:
        image: a ``PIL.Image`` (any mode; converted to RGB).
        phrase: the concept to localize (``"eyes"``, ``"the dress"``, …). May be
            a ``|``-separated list of phrases whose detections are unioned.
        threshold: minimum SAM3 score to keep an instance.
        dilate: grow the mask by this many pixels (margin for the fill); 0 = none.
        device: torch device string for the SAM3 model.
        checkpoint: SAM3 weights path (defaults to ``models/sam3/sam3.pt``).

    Returns:
        ``np.ndarray`` of shape ``(H, W)``, dtype ``uint8``, values ``{0, 1}`` —
        1 where the concept was detected (the region to edit). All-zero when the
        concept isn't found (caller falls back to an explicit mask).
    """
    import torch

    pil = image.convert("RGB")
    w, h = pil.size
    processor = _get_processor(device, checkpoint)

    out = np.zeros((h, w), dtype=np.uint8)
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.startswith("cuda")
        else torch.autocast(device_type="cpu", enabled=False)
    )
    with autocast:
        state = processor.set_image(pil)
        for sub in (p.strip() for p in phrase.split("|") if p.strip()):
            output = processor.set_text_prompt(state=state, prompt=sub)
            for mask, score in zip(output["masks"], output["scores"]):
                if float(score) < threshold:
                    continue
                m = mask.cpu().numpy() if torch.is_tensor(mask) else np.asarray(mask)
                if m.ndim == 3:
                    m = m[0]
                out = np.maximum(out, (m > 0.5).astype(np.uint8))

    if dilate > 0 and out.any():
        import cv2

        kernel = np.ones((dilate, dilate), dtype=np.uint8)
        out = cv2.dilate(out, kernel, iterations=1)
    return out
