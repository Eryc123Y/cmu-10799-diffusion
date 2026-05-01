"""Run the HW1 Part IV DDPM parameterization sweep on one Colab GPU.

The script launches independent train.py subprocesses for epsilon, x0,
velocity, and score prediction. Each subprocess writes to its own run
directory so Colab disconnects leave recoverable checkpoints and metrics.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = [
    REPO_ROOT / "configs" / "part4" / "ddpm_epsilon_colab.yaml",
    REPO_ROOT / "configs" / "part4" / "ddpm_x0_colab.yaml",
    REPO_ROOT / "configs" / "part4" / "ddpm_v_colab.yaml",
    REPO_ROOT / "configs" / "part4" / "ddpm_score_colab.yaml",
]


@dataclass
class RunRecord:
    prediction_type: str
    config_path: str
    run_dir: str
    config_used_path: str
    train_log_path: str
    final_checkpoint_path: str
    command: list[str]
    status: str = "pending"
    exit_code: int | None = None
    started_at: str | None = None
    finished_at: str | None = None
    resume_checkpoint: str | None = None
    error: str | None = None
    pid: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch Part IV DDPM parameterization training sweep."
    )
    parser.add_argument(
        "--configs",
        type=Path,
        nargs="+",
        default=DEFAULT_CONFIGS,
        help="Config files to train. Defaults to the four Part IV configs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/content/drive/MyDrive/cmu-10799-diffusion/part4_runs"),
    )
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--num-iterations", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--sample-every", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--method", choices=["ddpm"], default="ddpm")
    parser.add_argument("--python-executable", type=str, default=sys.executable)
    parser.add_argument("--resume-latest", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--status-every",
        type=int,
        default=30,
        help=(
            "Print a compact status update every N seconds by tailing each "
            "running train.log. Use 0 to disable periodic status output."
        ),
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as file:
        return yaml.safe_load(file)


def write_yaml(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        yaml.safe_dump(config, file, sort_keys=False)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def prediction_type_from_config(config: dict[str, Any]) -> str:
    prediction_type = config.get("ddpm", {}).get("prediction_type", "epsilon")
    if prediction_type not in {"epsilon", "x0", "v", "score"}:
        raise ValueError(f"Unsupported prediction_type in config: {prediction_type}")
    return prediction_type


def latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return None
    checkpoints = sorted(checkpoint_dir.glob("ddpm_*.pt"))
    numbered = [
        path
        for path in checkpoints
        if path.stem != "ddpm_final"
    ]
    if not numbered:
        return None
    return numbered[-1]


def apply_overrides(
    config: dict[str, Any],
    prediction_type: str,
    run_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    updated = dict(config)
    updated["training"] = dict(config["training"])
    updated["ddpm"] = dict(config["ddpm"])
    updated["logging"] = dict(config["logging"])

    if args.num_iterations is not None:
        updated["training"]["num_iterations"] = args.num_iterations
    if args.log_every is not None:
        updated["training"]["log_every"] = args.log_every
    if args.sample_every is not None:
        updated["training"]["sample_every"] = args.sample_every
    if args.save_every is not None:
        updated["training"]["save_every"] = args.save_every

    updated["ddpm"]["prediction_type"] = prediction_type
    updated["logging"]["dir"] = str(run_root)
    updated["logging"]["run_name"] = prediction_type
    updated["checkpoint"] = dict(config.get("checkpoint", {}))
    updated["checkpoint"]["resume"] = None
    return updated


def write_summary(run_root: Path, records: list[RunRecord]) -> None:
    summary = {
        "run_root": str(run_root),
        "updated_at": now_iso(),
        "runs": [asdict(record) for record in records],
    }
    summary_path = run_root / "sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))


def latest_log_line(log_path: str, max_bytes: int = 32768) -> str:
    """Read the latest non-empty line from a training log without loading it all."""
    path = Path(log_path)
    if not path.exists():
        return "log not created yet"

    with path.open("rb") as file:
        file.seek(0, 2)
        file_size = file.tell()
        file.seek(max(0, file_size - max_bytes))
        text = file.read().decode("utf-8", errors="replace")

    # tqdm writes carriage-return progress updates, so treat \\r as a line break.
    lines = [
        line.strip()
        for line in text.replace("\r", "\n").splitlines()
        if line.strip()
    ]
    if not lines:
        return "log is empty"
    return lines[-1][-240:]


def print_running_status(
    running: list[tuple[RunRecord, subprocess.Popen, Any]]
) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    names = ", ".join(record.prediction_type for record, _, _ in running)
    print(f"[status {timestamp}] running: {names}", flush=True)
    for record, _, _ in running:
        print(
            f"  - {record.prediction_type}: {latest_log_line(record.train_log_path)}",
            flush=True,
        )


def prepare_records(args: argparse.Namespace) -> tuple[Path, list[RunRecord]]:
    if args.max_parallel <= 0:
        raise ValueError(f"--max-parallel must be positive, got {args.max_parallel}")

    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = args.output_root / timestamp
    run_root.mkdir(parents=True, exist_ok=True)

    records: list[RunRecord] = []
    seen_prediction_types: set[str] = set()

    for config_path in args.configs:
        resolved_config_path = config_path.resolve()
        config = load_yaml(resolved_config_path)
        prediction_type = prediction_type_from_config(config)
        if prediction_type in seen_prediction_types:
            raise ValueError(f"Duplicate prediction_type in sweep: {prediction_type}")
        seen_prediction_types.add(prediction_type)

        run_dir = run_root / prediction_type
        if args.overwrite and run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        final_checkpoint = run_dir / "checkpoints" / f"{args.method}_final.pt"
        config_used_path = run_dir / "config_used.yaml"
        train_log_path = run_dir / "train.log"

        updated_config = apply_overrides(config, prediction_type, run_root, args)
        write_yaml(config_used_path, updated_config)

        command = [
            args.python_executable,
            str(REPO_ROOT / "train.py"),
            "--method",
            args.method,
            "--config",
            str(config_used_path),
        ]

        resume_checkpoint = None
        if args.resume_latest and not final_checkpoint.exists():
            latest = latest_checkpoint(run_dir)
            if latest is not None:
                resume_checkpoint = str(latest)
                command.extend(["--resume", resume_checkpoint])

        record = RunRecord(
            prediction_type=prediction_type,
            config_path=str(resolved_config_path),
            run_dir=str(run_dir),
            config_used_path=str(config_used_path),
            train_log_path=str(train_log_path),
            final_checkpoint_path=str(final_checkpoint),
            command=command,
            resume_checkpoint=resume_checkpoint,
        )
        if final_checkpoint.exists() and not args.overwrite:
            record.status = "skipped_existing_final"
            record.exit_code = 0
        records.append(record)

    return run_root, records


def run_sweep(run_root: Path, records: list[RunRecord], args: argparse.Namespace) -> None:
    pending = [record for record in records if record.status == "pending"]
    running: list[tuple[RunRecord, subprocess.Popen, Any]] = []
    last_status_at = 0.0

    if args.dry_run:
        for record in records:
            print(" ".join(record.command), flush=True)
        write_summary(run_root, records)
        return

    write_summary(run_root, records)

    while pending or running:
        while pending and len(running) < args.max_parallel:
            record = pending.pop(0)
            log_file = open(record.train_log_path, "a")
            record.status = "running"
            record.started_at = now_iso()
            print(
                f"[start] {record.prediction_type}: {' '.join(record.command)}",
                flush=True,
            )
            process = subprocess.Popen(
                record.command,
                cwd=REPO_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            record.pid = process.pid
            running.append((record, process, log_file))
            write_summary(run_root, records)

        time.sleep(5)
        if (
            running
            and args.status_every > 0
            and time.monotonic() - last_status_at >= args.status_every
        ):
            print_running_status(running)
            last_status_at = time.monotonic()

        still_running: list[tuple[RunRecord, subprocess.Popen, Any]] = []
        for record, process, log_file in running:
            exit_code = process.poll()
            if exit_code is None:
                still_running.append((record, process, log_file))
                continue

            log_file.close()
            record.exit_code = exit_code
            record.finished_at = now_iso()
            if exit_code == 0:
                record.status = "completed"
                print(f"[done] {record.prediction_type}", flush=True)
            else:
                record.status = "failed"
                record.error = f"train.py exited with code {exit_code}"
                print(
                    f"[failed] {record.prediction_type}: exit code {exit_code}",
                    flush=True,
                )
            write_summary(run_root, records)

        running = still_running

    write_summary(run_root, records)
    print(f"Sweep summary: {run_root / 'sweep_summary.json'}", flush=True)


def main() -> None:
    args = parse_args()
    run_root, records = prepare_records(args)
    print("=" * 72, flush=True)
    print("Part IV parameterization sweep", flush=True)
    print("=" * 72, flush=True)
    print(f"Run root: {run_root}", flush=True)
    print(f"Max parallel: {args.max_parallel}", flush=True)
    print(
        f"Runs: {', '.join(record.prediction_type for record in records)}",
        flush=True,
    )
    print("=" * 72, flush=True)
    run_sweep(run_root, records, args)


if __name__ == "__main__":
    main()
