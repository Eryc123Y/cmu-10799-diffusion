#!/usr/bin/env python3
"""Lecture 4 score-scale visualizer.

This script computes the DDPM score-target scale

    score_scale_t = 1 / sqrt(1 - alpha_bar_t)

for a linear beta schedule. It is meant for the mini coding task in
exercise/lecture-4-score/lecture4-score-exercises.typ.
"""

from __future__ import annotations

import argparse
import math
from typing import Iterable


def score_scale_table(
    num_timesteps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> list[tuple[int, float, float]]:
    """Return (t, sqrt_one_minus_alpha_bar_t, score_scale_t)."""
    if num_timesteps <= 0:
        raise ValueError("num_timesteps must be positive")
    if beta_start <= 0 or beta_end <= 0:
        raise ValueError("beta_start and beta_end must be positive")

    alpha_bar = 1.0
    rows: list[tuple[int, float, float]] = []

    for t in range(num_timesteps):
        if num_timesteps == 1:
            beta_t = beta_start
        else:
            beta_t = beta_start + (beta_end - beta_start) * t / (num_timesteps - 1)

        alpha_t = 1.0 - beta_t
        alpha_bar *= alpha_t

        sqrt_one_minus_alpha_bar_t = math.sqrt(1.0 - alpha_bar)
        score_scale_t = 1.0 / sqrt_one_minus_alpha_bar_t
        rows.append((t, sqrt_one_minus_alpha_bar_t, score_scale_t))

    return rows


def print_selected_rows(
    rows: list[tuple[int, float, float]],
    timesteps: Iterable[int],
) -> None:
    """Print selected timesteps as a compact table."""
    print(f"{'t':>5}  {'sqrt(1-alpha_bar_t)':>24}  {'1/sqrt(1-alpha_bar_t)':>26}")
    print("-" * 61)
    for t in timesteps:
        if t < 0 or t >= len(rows):
            print(f"{t:>5}  {'<out of range>':>24}  {'<out of range>':>26}")
            continue
        _, sigma_t, score_scale_t = rows[t]
        print(f"{t:5d}  {sigma_t:24.10f}  {score_scale_t:26.10f}")


def parse_timesteps(text: str) -> list[int]:
    """Parse a comma-separated timestep list such as '0,1,9,99,499,999'."""
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Print DDPM score target scales.")
    parser.add_argument("--num-timesteps", type=int, default=1000)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=0.02)
    parser.add_argument("--timesteps", type=str, default="0,1,9,99,499,999")
    args = parser.parse_args()

    rows = score_scale_table(
        num_timesteps=args.num_timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
    )
    print_selected_rows(rows, parse_timesteps(args.timesteps))


if __name__ == "__main__":
    main()
