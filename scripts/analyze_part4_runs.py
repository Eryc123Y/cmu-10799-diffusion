"""Create report-ready figures from HW1 Part IV sweep outputs."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

import json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Part IV sweep outputs.")
    parser.add_argument("--sweep-summary", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, default=None)
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("HW1_CMU_10799_Spring_2026/figures"),
    )
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        print(f"Missing CSV: {path}")
        return []
    with path.open("r", newline="") as file:
        return list(csv.DictReader(file))


def to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def finite_xy(rows: list[dict[str, str]], y_key: str) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    values: list[float] = []
    for row in rows:
        step = int(float(row["step"]))
        value = to_float(row.get(y_key, "nan"))
        if math.isfinite(value):
            steps.append(step)
            values.append(value)
    return steps, values


def successful_runs(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        run
        for run in summary["runs"]
        if run.get("status") in {"completed", "skipped_existing_final"}
    ]


def plot_metric_curves(
    runs: list[dict[str, Any]],
    metric: str,
    ylabel: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 4.8))
    plotted = False
    for run in runs:
        metrics_path = Path(run["run_dir"]) / "metrics.csv"
        rows = read_csv(metrics_path)
        steps, values = finite_xy(rows, metric)
        if not steps:
            continue
        plt.plot(steps, values, label=run["prediction_type"], linewidth=1.8)
        plotted = True

    if not plotted:
        print(f"No data available for {metric}; skipping {output_path}")
        plt.close()
        return

    plt.xlabel("Training step")
    plt.ylabel(ylabel)
    plt.title(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"Wrote figure: {output_path}")


def plot_timestep_heatmap(runs: list[dict[str, Any]], output_path: Path) -> None:
    bin_columns = [f"bin_{i * 100:03d}_{(i + 1) * 100 - 1:03d}_noise_mse" for i in range(9)]
    bin_columns.append("bin_900_999_noise_mse")

    labels: list[str] = []
    values: list[list[float]] = []
    for run in runs:
        rows = read_csv(Path(run["run_dir"]) / "metrics.csv")
        if not rows:
            continue
        last_row = rows[-1]
        row_values = [to_float(last_row.get(column, "nan")) for column in bin_columns]
        if any(math.isfinite(value) for value in row_values):
            labels.append(run["prediction_type"])
            values.append(row_values)

    if not values:
        print(f"No timestep-bin data; skipping {output_path}")
        return

    plt.figure(figsize=(8, 3.4))
    plt.imshow(values, aspect="auto", cmap="viridis")
    plt.colorbar(label="Noise MSE")
    plt.yticks(range(len(labels)), labels)
    plt.xticks(range(10), [f"{i * 100}-{i * 100 + 99}" for i in range(10)], rotation=35)
    plt.xlabel("Timestep bin")
    plt.title("Final logged noise MSE by timestep bin")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"Wrote figure: {output_path}")


def latest_sample_grid(run_dir: Path) -> Path | None:
    sample_dir = run_dir / "samples"
    if not sample_dir.exists():
        return None
    samples = sorted(sample_dir.glob("samples_*.png"))
    if not samples:
        return None
    return samples[-1]


def labeled_tile(image_path: Path, label: str, width: int = 320) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    scale = width / image.width
    height = max(1, int(image.height * scale))
    image = image.resize((width, height))
    title_height = 32
    canvas = Image.new("RGB", (width, height + title_height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), label, fill="black")
    canvas.paste(image, (0, title_height))
    return canvas


def combine_tiles(tiles: list[Image.Image], output_path: Path, columns: int = 2) -> None:
    if not tiles:
        print(f"No image tiles; skipping {output_path}")
        return
    tile_width = max(tile.width for tile in tiles)
    tile_height = max(tile.height for tile in tiles)
    rows = math.ceil(len(tiles) / columns)
    canvas = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
    for index, tile in enumerate(tiles):
        row = index // columns
        column = index % columns
        canvas.paste(tile, (column * tile_width, row * tile_height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(f"Wrote figure: {output_path}")


def make_q6_sample_grid(runs: list[dict[str, Any]], output_path: Path) -> None:
    tiles = []
    for run in runs:
        sample_path = latest_sample_grid(Path(run["run_dir"]))
        if sample_path is None:
            print(f"No sample grid found for {run['prediction_type']}")
            continue
        tiles.append(labeled_tile(sample_path, run["prediction_type"], width=300))
    combine_tiles(tiles, output_path, columns=2)


def make_q6_trajectory_grid(eval_dir: Path, runs: list[dict[str, Any]], output_path: Path) -> None:
    tiles = []
    for run in runs:
        prediction_type = run["prediction_type"]
        trajectory_path = (
            eval_dir
            / "q6_parameterization"
            / prediction_type
            / f"{prediction_type}_reverse_trajectory.png"
        )
        if not trajectory_path.exists():
            print(f"No trajectory found for {prediction_type}: {trajectory_path}")
            continue
        tiles.append(labeled_tile(trajectory_path, prediction_type, width=520))
    combine_tiles(tiles, output_path, columns=1)


def first_png(directory: Path) -> Path | None:
    pngs = sorted(directory.glob("*.png"))
    if not pngs:
        return None
    return pngs[0]


def make_q7_sample_grid(q7_table: Path, output_path: Path) -> None:
    rows = read_csv(q7_table)
    tiles = []
    for row in rows:
        generated_dir = Path(row["generated_dir"])
        image_path = first_png(generated_dir)
        if image_path is None:
            print(f"No generated PNG found in {generated_dir}")
            continue
        tiles.append(labeled_tile(image_path, f"{row['num_steps']} steps", width=160))
    combine_tiles(tiles, output_path, columns=6)


def plot_q7_kid(q7_table: Path, output_path: Path) -> None:
    rows = read_csv(q7_table)
    steps = []
    means = []
    stds = []
    for row in rows:
        step = int(float(row["num_steps"]))
        mean = to_float(row["kernel_inception_distance_mean"])
        std = to_float(row["kernel_inception_distance_std"])
        if math.isfinite(mean):
            steps.append(step)
            means.append(mean)
            stds.append(std if math.isfinite(std) else 0.0)

    if not steps:
        print(f"No Q7 KID rows; skipping {output_path}")
        return

    plt.figure(figsize=(7, 4.4))
    plt.errorbar(steps, means, yerr=stds, marker="o", capsize=3, linewidth=1.8)
    plt.xlabel("Sampling steps")
    plt.ylabel("KID")
    plt.title("KID vs sampling steps")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"Wrote figure: {output_path}")


def copy_table(source: Path, destination: Path) -> None:
    if not source.exists():
        print(f"Missing table: {source}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    print(f"Copied table: {destination}")


def main() -> None:
    args = parse_args()
    summary = load_summary(args.sweep_summary)
    run_root = Path(summary["run_root"])
    eval_dir = args.evaluation_dir if args.evaluation_dir is not None else run_root / "evaluation"
    figures_dir = args.figures_dir
    runs = successful_runs(summary)

    plot_metric_curves(
        runs,
        metric="loss",
        ylabel="Training loss",
        output_path=figures_dir / "q6_loss_curves.png",
    )
    plot_metric_curves(
        runs,
        metric="noise_mse",
        ylabel="Noise MSE",
        output_path=figures_dir / "q6_noise_mse_curves.png",
    )
    plot_timestep_heatmap(runs, figures_dir / "q6_timestep_bin_heatmap.png")
    make_q6_sample_grid(runs, figures_dir / "q6_final_samples.png")
    make_q6_trajectory_grid(eval_dir, runs, figures_dir / "q6_reverse_trajectories.png")

    q6_table = eval_dir / "q6_parameterization" / "q6_kid_table.csv"
    q7_table = eval_dir / "q7_sampling_steps" / "q7_kid_table.csv"
    copy_table(q6_table, figures_dir / "q6_kid_table.csv")
    copy_table(q7_table, figures_dir / "q7_kid_table.csv")
    make_q7_sample_grid(q7_table, figures_dir / "q7_sampling_steps_samples.png")
    plot_q7_kid(q7_table, figures_dir / "q7_kid_vs_steps.png")

    print("=" * 72)
    print("Part IV analysis complete")
    print(f"Figures dir: {figures_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
