"""HuggingFace Accelerate setup and BF16 plumbing.

Wraps the ``Accelerator`` construction so training scripts can stay out of the
logging-backend plumbing. Also hosts the state-resume helper (local dir or HF
repo) and the dtype resolver (weights and checkpoints are always bf16).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
from typing import Optional

import torch
from accelerate import Accelerator
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)


def resume_from_local_or_hf_if_specified(accelerator, args):
    if not args.resume:
        return

    if not args.resume_from_huggingface:
        logger.info(f"resume training from local state: {args.resume}")
        accelerator.load_state(args.resume)
        return

    logger.info(f"resume training from huggingface state: {args.resume}")
    repo_id = args.resume.split("/")[0] + "/" + args.resume.split("/")[1]
    path_in_repo = "/".join(args.resume.split("/")[2:])
    revision = None
    repo_type = None
    if ":" in path_in_repo:
        divided = path_in_repo.split(":")
        if len(divided) == 2:
            path_in_repo, revision = divided
            repo_type = "model"
        else:
            path_in_repo, revision, repo_type = divided
    logger.info(
        f"Downloading state from huggingface: {repo_id}/{path_in_repo}@{revision}"
    )

    from huggingface_hub import list_repo_files

    list_files = list_repo_files(
        repo_id=repo_id,
        revision=revision,
        token=args.huggingface_token,
        repo_type=repo_type,
    )
    list_files = [f for f in list_files if f.startswith(path_in_repo)]

    async def download(filename) -> str:
        def task():
            return hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                repo_type=repo_type,
                token=args.huggingface_token,
            )

        return await asyncio.get_event_loop().run_in_executor(None, task)

    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(
        asyncio.gather(*[download(filename=filename) for filename in list_files])
    )
    if len(results) == 0:
        raise ValueError("No files found in the specified repo id")
    dirname = os.path.dirname(results[0])
    accelerator.load_state(dirname)


def resolve_run_log_dir(args: argparse.Namespace) -> Optional[str]:
    """Return the timestamped run-specific log directory for this invocation.

    Builds ``{args.logging_dir}/{log_prefix}{YYYYMMDD-HHMM}`` once and caches it
    on ``args._resolved_run_log_dir`` so the config-snapshot writer (which runs
    before ``prepare_accelerator``) and the accelerator setup land in the same
    dir even when they're a minute apart on the wall clock. Returns ``None``
    when ``--logging_dir`` is unset.
    """
    if args.logging_dir is None:
        return None
    cached = getattr(args, "_resolved_run_log_dir", None)
    if cached is not None:
        return cached
    if args.log_prefix is not None:
        log_prefix = args.log_prefix
    else:
        # Prefer output_name so TensorBoard run dirs match the GUI run title.
        output_name = getattr(args, "output_name", None)
        if output_name:
            log_prefix = output_name + "_"
        else:
            method = getattr(args, "method", None)
            preset = getattr(args, "preset", None)
            parts = [p for p in (method, preset) if p]
            log_prefix = ("_".join(parts) + "_") if parts else ""
    resolved = (
        args.logging_dir
        + "/"
        + log_prefix
        + time.strftime("%Y%m%d-%H%M", time.localtime())
    )
    args._resolved_run_log_dir = resolved
    return resolved


def prepare_accelerator(args: argparse.Namespace):
    logging_dir = resolve_run_log_dir(args)

    if args.log_with is None:
        if logging_dir is not None:
            log_with = "tensorboard"
        else:
            log_with = None
    else:
        log_with = args.log_with
        if log_with in ["tensorboard", "all"]:
            if logging_dir is None:
                raise ValueError("logging_dir is required when log_with is tensorboard")
        if log_with in ["wandb", "all"]:
            try:
                import wandb
            except ImportError:
                raise ImportError("No wandb")
            if logging_dir is not None:
                os.makedirs(logging_dir, exist_ok=True)
                os.environ["WANDB_DIR"] = logging_dir
            if args.wandb_api_key is not None:
                wandb.login(key=args.wandb_api_key)

    # Always "NO": torch.compile is applied per-block by DiT.compile_blocks (the
    # block loop is the shape-stable hot path; pre/post-block regions vary per
    # bucket). Letting Accelerate full-compile the model on top of that would
    # double-compile and trigger graph-break / recompile storms.
    dynamo_backend = "NO"

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",  # Anima trains in bf16, always
        log_with=log_with,
        project_dir=logging_dir,
        dynamo_backend=dynamo_backend,
    )
    print("accelerator device:", accelerator.device)
    return accelerator


def prepare_dtype(args: argparse.Namespace):
    # Anima trains and saves in bf16, always.
    weight_dtype = save_dtype = torch.bfloat16
    return weight_dtype, save_dtype
