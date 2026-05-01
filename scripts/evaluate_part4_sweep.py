"""Evaluate HW1 Part IV sweep checkpoints with KID and sampling-step ablations."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from torchvision.utils import save_image as torchvision_save_image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sample import load_checkpoint
from src.data import unnormalize
from src.methods import DDPM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Part IV parameterization sweep and step ablation."
    )
    parser.add_argument("--sweep-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--sample-batch-size", type=int, default=32)
    parser.add_argument("--real-batch-size", type=int, default=64)
    parser.add_argument("--real-num-workers", type=int, default=0)
    parser.add_argument("--kid-subset-size", type=int, default=100)
    parser.add_argument("--kid-subsets", type=int, default=10)
    parser.add_argument("--steps", type=int, nargs="+", default=[100, 300, 500, 700, 900, 1000])
    parser.add_argument("--baseline-prediction-type", type=str, default="epsilon")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-q6", action="store_true")
    parser.add_argument("--skip-q7", action="store_true")
    parser.add_argument("--no-trajectories", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested_device)


def successful_runs(summary: dict[str, Any]) -> list[dict[str, Any]]:
    runs = []
    for run in summary["runs"]:
        status = run.get("status")
        checkpoint = Path(run["final_checkpoint_path"])
        if status in {"completed", "skipped_existing_final"} and checkpoint.exists():
            runs.append(run)
        else:
            print(
                "Skipping run without final checkpoint: "
                f"{run.get('prediction_type')} status={status}"
            )
    return runs


def run_evaluate_kid(
    checkpoint: Path,
    generated_dir: Path,
    real_dir: Path,
    metrics_output: Path,
    args: argparse.Namespace,
    num_steps: int | None = None,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "evaluate_kid.py"),
        "--checkpoint",
        str(checkpoint),
        "--generated-dir",
        str(generated_dir),
        "--real-dir",
        str(real_dir),
        "--metrics-output",
        str(metrics_output),
        "--num-samples",
        str(args.num_samples),
        "--sample-batch-size",
        str(args.sample_batch_size),
        "--real-batch-size",
        str(args.real_batch_size),
        "--real-num-workers",
        str(args.real_num_workers),
        "--kid-subset-size",
        str(args.kid_subset_size),
        "--kid-subsets",
        str(args.kid_subsets),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
    ]
    if num_steps is not None:
        command.extend(["--num-steps", str(num_steps)])

    print("=" * 72)
    print("Running KID evaluation")
    print(" ".join(command))
    print("=" * 72)
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    return json.loads(metrics_output.read_text())


def kid_row(
    *,
    prediction_type: str,
    checkpoint: Path,
    generated_dir: Path,
    real_dir: Path,
    metrics_output: Path,
    metrics: dict[str, Any],
    num_steps: int | None = None,
    reused_from: str | None = None,
) -> dict[str, Any]:
    return {
        "prediction_type": prediction_type,
        "num_steps": "" if num_steps is None else num_steps,
        "checkpoint": str(checkpoint),
        "generated_dir": str(generated_dir),
        "real_dir": str(real_dir),
        "metrics_output": str(metrics_output),
        "kernel_inception_distance_mean": metrics["kernel_inception_distance_mean"],
        "kernel_inception_distance_std": metrics["kernel_inception_distance_std"],
        "reused_from": reused_from or "",
    }


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote table: {path}")


@torch.no_grad()
def save_trajectories(
    checkpoint: Path,
    prediction_type: str,
    output_dir: Path,
    device: torch.device,
    seed: int,
    selected_steps: tuple[int, ...] = (999, 700, 500, 300, 100, 0),
) -> None:
    """Save x_t and predicted x_0 trajectories for one checkpoint."""
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model, config, ema = load_checkpoint(str(checkpoint), device)
    method = DDPM.from_config(model, config, device)
    ema.apply_shadow()
    method.eval_mode()

    data_config = config["data"]
    image_shape = (
        data_config["channels"],
        data_config["image_size"],
        data_config["image_size"],
    )
    x_t = torch.randn((1, *image_shape), device=device)
    xt_images: list[torch.Tensor] = []
    x0_images: list[torch.Tensor] = []

    selected = set(selected_steps)
    for step in reversed(range(method.num_timesteps)):
        t = torch.full((1,), step, device=device, dtype=torch.long)
        if step in selected:
            model_output = method.model(x_t, t)
            pred_x0 = method._prediction_to_x0(x_t, t, model_output)
            xt_images.append(x_t.detach().cpu())
            x0_images.append(pred_x0.detach().cpu())
        x_t = method.reverse_process(x_t, t)

    xt_grid = unnormalize(torch.cat(xt_images, dim=0)).clamp(0.0, 1.0)
    x0_grid = unnormalize(torch.cat(x0_images, dim=0)).clamp(0.0, 1.0)
    torchvision_save_image(
        xt_grid,
        output_dir / f"{prediction_type}_reverse_trajectory.png",
        nrow=len(xt_images),
    )
    torchvision_save_image(
        x0_grid,
        output_dir / f"{prediction_type}_x0_prediction_trajectory.png",
        nrow=len(x0_images),
    )
    ema.restore()


def evaluate_q6(
    runs: list[dict[str, Any]],
    eval_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    q6_dir = eval_dir / "q6_parameterization"
    real_dir = eval_dir / "real_1k"
    rows = []
    results: dict[str, dict[str, Any]] = {}

    for run in runs:
        prediction_type = run["prediction_type"]
        checkpoint = Path(run["final_checkpoint_path"])
        run_eval_dir = q6_dir / prediction_type
        generated_dir = run_eval_dir / "generated_1k"
        metrics_output = run_eval_dir / "kid_metrics.json"

        metrics = run_evaluate_kid(
            checkpoint=checkpoint,
            generated_dir=generated_dir,
            real_dir=real_dir,
            metrics_output=metrics_output,
            args=args,
        )
        row = kid_row(
            prediction_type=prediction_type,
            checkpoint=checkpoint,
            generated_dir=generated_dir,
            real_dir=real_dir,
            metrics_output=metrics_output,
            metrics=metrics,
        )
        rows.append(row)
        results[prediction_type] = row

        if not args.no_trajectories:
            save_trajectories(
                checkpoint=checkpoint,
                prediction_type=prediction_type,
                output_dir=run_eval_dir,
                device=device,
                seed=args.seed,
            )

    write_table(q6_dir / "q6_kid_table.csv", rows)
    return results


def evaluate_q7(
    runs: list[dict[str, Any]],
    eval_dir: Path,
    args: argparse.Namespace,
    q6_results: dict[str, dict[str, Any]],
) -> None:
    baseline = None
    for run in runs:
        if run["prediction_type"] == args.baseline_prediction_type:
            baseline = run
            break
    if baseline is None:
        raise RuntimeError(
            f"No successful baseline run found for {args.baseline_prediction_type!r}"
        )

    checkpoint = Path(baseline["final_checkpoint_path"])
    q7_dir = eval_dir / "q7_sampling_steps"
    real_dir = eval_dir / "real_1k"
    rows = []

    for num_steps in args.steps:
        if num_steps == 1000 and args.baseline_prediction_type in q6_results:
            reused = q6_results[args.baseline_prediction_type]
            metrics = json.loads(Path(reused["metrics_output"]).read_text())
            rows.append(
                kid_row(
                    prediction_type=args.baseline_prediction_type,
                    checkpoint=checkpoint,
                    generated_dir=Path(reused["generated_dir"]),
                    real_dir=Path(reused["real_dir"]),
                    metrics_output=Path(reused["metrics_output"]),
                    metrics=metrics,
                    num_steps=num_steps,
                    reused_from="q6_baseline_1000_steps",
                )
            )
            continue

        step_dir = q7_dir / f"steps_{num_steps:04d}"
        generated_dir = step_dir / "generated_1k"
        metrics_output = step_dir / f"kid_{num_steps:04d}.json"
        metrics = run_evaluate_kid(
            checkpoint=checkpoint,
            generated_dir=generated_dir,
            real_dir=real_dir,
            metrics_output=metrics_output,
            args=args,
            num_steps=num_steps,
        )
        rows.append(
            kid_row(
                prediction_type=args.baseline_prediction_type,
                checkpoint=checkpoint,
                generated_dir=generated_dir,
                real_dir=real_dir,
                metrics_output=metrics_output,
                metrics=metrics,
                num_steps=num_steps,
            )
        )

    rows.sort(key=lambda row: int(row["num_steps"]))
    write_table(q7_dir / "q7_kid_table.csv", rows)


def main() -> None:
    args = parse_args()
    summary = load_summary(args.sweep_summary)
    run_root = Path(summary["run_root"])
    eval_dir = args.output_dir if args.output_dir is not None else run_root / "evaluation"
    if args.overwrite and eval_dir.exists():
        shutil.rmtree(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    runs = successful_runs(summary)
    if not runs:
        raise RuntimeError("No successful runs with final checkpoints were found.")

    device = resolve_device(args.device)
    q6_results: dict[str, dict[str, Any]] = {}
    if not args.skip_q6:
        q6_results = evaluate_q6(runs, eval_dir, args, device)
    if not args.skip_q7:
        evaluate_q7(runs, eval_dir, args, q6_results)

    print("=" * 72)
    print("Part IV evaluation complete")
    print(f"Evaluation dir: {eval_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
