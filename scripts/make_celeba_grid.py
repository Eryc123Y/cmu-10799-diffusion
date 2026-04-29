"""Create a random CelebA sample grid for HW1 Part I."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import CelebADataset, save_image, unnormalize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a random CelebA sample grid.")
    parser.add_argument("--root", type=str, default="./data/celeba-subset")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--output",
        type=str,
        default="HW1_CMU_10799_Spring_2026/figures/part_i_samples.png",
    )
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--from-hub", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = CelebADataset(
        root=args.root,
        split=args.split,
        image_size=args.image_size,
        augment=False,
        from_hub=args.from_hub,
    )

    if len(dataset) < args.num_samples:
        raise ValueError(
            f"Dataset has only {len(dataset)} samples, but {args.num_samples} were requested."
        )

    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(dataset), generator=generator)[: args.num_samples].tolist()
    images = torch.stack([dataset[index] for index in indices], dim=0)

    if images.ndim != 4 or images.shape[1:] != (3, args.image_size, args.image_size):
        raise RuntimeError(f"Unexpected image batch shape: {tuple(images.shape)}")
    if not torch.isfinite(images).all():
        raise RuntimeError("Image batch contains NaN or Inf values.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nrow = int(math.sqrt(args.num_samples))
    save_image(unnormalize(images).clamp(0.0, 1.0), str(output_path), nrow=nrow)

    print(f"Saved {args.num_samples} samples to {output_path}")
    print(f"Tensor shape: {tuple(images.shape)}")
    print(f"Tensor range: [{images.min().item():.3f}, {images.max().item():.3f}]")


if __name__ == "__main__":
    main()
