from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Force CPU-only test runs unless explicitly opted into GPU. Several code paths
# select their device from ``torch.cuda.is_available()`` (LoRA / chimera SVD
# init, network assembly, …), so a card present on the box silently pulls
# tensors onto the GPU. Hiding the device here — before torch is imported
# anywhere — keeps the suite deterministic and identical with or without a GPU.
# Set ANIMA_TEST_GPU=1 to allow the GPU back in.
if os.environ.get("ANIMA_TEST_GPU") != "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session", autouse=True)
def _chdir_repo_root(repo_root: Path):
    prev = os.getcwd()
    os.chdir(repo_root)
    try:
        yield
    finally:
        os.chdir(prev)


@pytest.fixture
def tiny_args() -> argparse.Namespace:
    return argparse.Namespace(
        method="lora",
        preset="default",
        learning_rate=1e-4,
        network_dim=4,
        network_alpha=4,
        seed=42,
    )


@pytest.fixture
def tiny_latents():
    import torch

    torch.manual_seed(0)
    return torch.randn(1, 16, 8, 8, dtype=torch.float32)


def iter_method_names():
    methods_dir = REPO_ROOT / "configs" / "methods"
    for path in sorted(methods_dir.glob("*.toml")):
        yield path.stem
