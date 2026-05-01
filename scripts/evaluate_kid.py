"""Generate DDPM samples and evaluate KID against CelebA subset images.

This script is intentionally plain Python so the Colab notebook only needs to
run one top-level command. It avoids notebook-side Drive/upload logic.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from torchvision.utils import save_image as torchvision_save_image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sample import load_checkpoint, save_samples
from src.data import create_dataloader, unnormalize
from src.methods import DDPM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 1k DDPM samples and compute KID."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ddpm_colab.yaml"),
        help="Kept for command readability; model/data config comes from checkpoint.",
    )
    parser.add_argument(
        "--generated-dir",
        type=Path,
        default=Path("/content/kid_eval/generated_1k"),
    )
    parser.add_argument(
        "--real-dir",
        type=Path,
        default=Path("/content/kid_eval/real_1k"),
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("/content/kid_eval/kid_metrics.json"),
    )
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--sample-batch-size", type=int, default=32)
    parser.add_argument("--real-batch-size", type=int, default=64)
    parser.add_argument(
        "--real-num-workers",
        type=int,
        default=0,
        help=(
            "DataLoader workers for exporting real images. The default 0 avoids "
            "notebook/local multiprocessing pickling issues during evaluation."
        ),
    )
    parser.add_argument("--kid-subset-size", type=int, default=100)
    parser.add_argument("--kid-subsets", type=int, default=10)
    parser.add_argument("--method", choices=["ddpm"], default="ddpm")
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--from-hub", action="store_true", default=None)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested_device)


def count_pngs(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for _ in directory.glob("*.png"))


def ensure_checkpoint(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "In Colab, upload ddpm_final.pt to /content/ddpm_final.pt or pass "
            "--checkpoint with the path where you placed it."
        )
    if path.stat().st_size < 10_000_000:
        raise ValueError(f"Checkpoint looks too small to be valid: {path}")


def generate_samples(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    if args.overwrite and args.generated_dir.exists():
        shutil.rmtree(args.generated_dir)
    args.generated_dir.mkdir(parents=True, exist_ok=True)

    existing = count_pngs(args.generated_dir)
    if existing >= args.num_samples:
        print(f"Reusing {existing} generated PNGs from {args.generated_dir}")
        return {"generated": existing, "skipped": True}

    print(f"Loading checkpoint from {args.checkpoint}")
    model, config, ema = load_checkpoint(str(args.checkpoint), device)

    if args.method == "ddpm":
        method = DDPM.from_config(model, config, device)
    else:
        raise ValueError(f"Unsupported method: {args.method}")

    if args.no_ema:
        print("Using raw training weights.")
    else:
        print("Using EMA weights.")
        ema.apply_shadow()

    method.eval_mode()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    data_config = config["data"]
    image_shape = (
        data_config["channels"],
        data_config["image_size"],
        data_config["image_size"],
    )
    num_steps = args.num_steps or config["sampling"]["num_steps"]
    sample_index = existing
    remaining = args.num_samples - existing

    print(
        "Generating "
        f"{remaining} samples into {args.generated_dir} "
        f"(batch_size={args.sample_batch_size}, num_steps={num_steps})"
    )
    with torch.no_grad():
        pbar = tqdm(total=remaining, desc="Generating samples")
        while remaining > 0:
            batch_size = min(args.sample_batch_size, remaining)
            samples = method.sample(
                batch_size=batch_size,
                image_shape=image_shape,
                num_steps=num_steps,
            )
            for image_idx in range(samples.shape[0]):
                output_path = args.generated_dir / f"{sample_index:06d}.png"
                save_samples(samples[image_idx : image_idx + 1], str(output_path), 1)
                sample_index += 1
            remaining -= batch_size
            pbar.update(batch_size)
        pbar.close()

    if not args.no_ema:
        ema.restore()

    total = count_pngs(args.generated_dir)
    print(f"Generated PNG count: {total}")
    return {"generated": total, "skipped": False}


def export_real_images(
    args: argparse.Namespace,
    checkpoint_config: dict[str, Any],
) -> dict[str, Any]:
    if args.overwrite and args.real_dir.exists():
        shutil.rmtree(args.real_dir)
    args.real_dir.mkdir(parents=True, exist_ok=True)

    existing = count_pngs(args.real_dir)
    if existing >= args.num_samples:
        print(f"Reusing {existing} real PNGs from {args.real_dir}")
        return {"real": existing, "skipped": True}

    if existing:
        shutil.rmtree(args.real_dir)
        args.real_dir.mkdir(parents=True, exist_ok=True)

    data_config = checkpoint_config["data"].copy()
    root = (
        args.data_root
        if args.data_root is not None
        else data_config.get("root", "./data/celeba-subset")
    )
    from_hub = data_config.get("from_hub", False)
    if args.from_hub is not None:
        from_hub = args.from_hub

    print(
        "Exporting real images from CelebA subset "
        f"(root={root}, from_hub={from_hub})"
    )
    loader = create_dataloader(
        root=root,
        split=args.split,
        image_size=data_config["image_size"],
        batch_size=args.real_batch_size,
        num_workers=args.real_num_workers,
        pin_memory=data_config.get("pin_memory", True),
        augment=False,
        shuffle=False,
        drop_last=False,
        from_hub=from_hub,
        repo_name=data_config.get(
            "repo_name",
            "electronickale/cmu-10799-celeba64-subset",
        ),
    )

    image_count = 0
    for batch in tqdm(loader, desc="Exporting real images"):
        images = unnormalize(batch).clamp(0.0, 1.0)
        for image in images:
            torchvision_save_image(image, args.real_dir / f"{image_count:06d}.png")
            image_count += 1
            if image_count >= args.num_samples:
                break
        if image_count >= args.num_samples:
            break

    print(f"Real PNG count: {image_count}")
    return {"real": image_count, "skipped": False}


def compute_kid(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    try:
        from torch_fidelity import calculate_metrics
    except ImportError as err:
        raise ImportError(
            "torch-fidelity is required. Install with: pip install torch-fidelity"
        ) from err

    generated_count = count_pngs(args.generated_dir)
    real_count = count_pngs(args.real_dir)
    if generated_count < args.num_samples:
        raise RuntimeError(
            f"Only {generated_count} generated images found in {args.generated_dir}; "
            f"expected {args.num_samples}."
        )
    if real_count < args.num_samples:
        raise RuntimeError(
            f"Only {real_count} real images found in {args.real_dir}; "
            f"expected {args.num_samples}."
        )

    print("Computing KID with torch-fidelity")
    metrics = calculate_metrics(
        input1=str(args.generated_dir),
        input2=str(args.real_dir),
        cuda=device.type == "cuda",
        isc=False,
        fid=False,
        kid=True,
        kid_subset_size=args.kid_subset_size,
        kid_subsets=args.kid_subsets,
        verbose=True,
    )
    return dict(metrics)


def load_checkpoint_config(checkpoint_path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return checkpoint["config"]


def write_metrics(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        key: value.item() if hasattr(value, "item") else value
        for key, value in metrics.items()
    }
    path.write_text(json.dumps(serializable, indent=2, sort_keys=True))
    print(f"Wrote metrics to {path}")


def main() -> None:
    args = parse_args()
    ensure_checkpoint(args.checkpoint)
    device = resolve_device(args.device)
    checkpoint_config = load_checkpoint_config(args.checkpoint)

    print("=" * 72)
    print("KID evaluation")
    print("=" * 72)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Generated dir: {args.generated_dir}")
    print(f"Real dir: {args.real_dir}")
    print(f"Num samples: {args.num_samples}")
    print(f"Sample batch size: {args.sample_batch_size}")
    print(f"KID subsets: {args.kid_subsets} x {args.kid_subset_size}")
    print("=" * 72)

    generate_samples(args, device)
    export_real_images(args, checkpoint_config)
    metrics = compute_kid(args, device)
    write_metrics(args.metrics_output, metrics)

    print("=" * 72)
    print("KID results")
    print("=" * 72)
    print(metrics)
    print("KID mean:", metrics["kernel_inception_distance_mean"])
    print("KID std:", metrics["kernel_inception_distance_std"])


if __name__ == "__main__":
    main()
