#!/usr/bin/env python3
"""Group dataset images by PE-Spatial visual similarity → groups.json manifest.

A curation tool (not a preprocess/training step): clusters near-identical /
same-concept images per artist so the GUI Dataset tab can filter by group and
near-duplicates are easy to spot. Argparse shell over
``library.datasets.grouping.build_groups``; driven by ``make curate-group``
(paths resolved from the config chain). Reuses the shared PE-Spatial feature
cache, so a re-run — or a re-run at a different ``--threshold`` — is cheap.
"""

import argparse
from pathlib import Path

from library.datasets.grouping import DEFAULT_THRESHOLD, build_groups


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source-dir", default="image_dataset", help="Native source image tree"
    )
    p.add_argument(
        "--out",
        default="post_image_dataset/groups/groups.json",
        help="Manifest path the GUI Dataset tab reads",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="cosine >= this connects two images (higher = tighter near-dups)",
    )
    p.add_argument(
        "--min-size",
        type=int,
        default=2,
        help="drop groups smaller than this (1 keeps singletons)",
    )
    p.add_argument("--encoder", default="pe_spatial", help="PE encoder name")
    p.add_argument("--batch-size", type=int, default=16, help="PE embed batch size")
    p.add_argument(
        "--num-workers", type=int, default=4, help="DataLoader image-decode workers"
    )
    p.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    args = p.parse_args()

    m = build_groups(
        Path(args.source_dir),
        Path(args.out),
        threshold=args.threshold,
        min_size=args.min_size,
        encoder=args.encoder,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(
        f"{m['n_groups']} group(s) over {m['n_images']} image(s) "
        f"({m['n_grouped']} grouped, {m['n_singletons']} ungrouped) "
        f"@ threshold {args.threshold} → {args.out}"
    )


if __name__ == "__main__":
    main()
