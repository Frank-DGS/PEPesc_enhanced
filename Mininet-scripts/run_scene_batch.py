#!/usr/bin/env python3
import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path


def load_scene_table(path: Path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"scene_id", "bw_mbps", "loss_pct", "duration_s"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Missing columns in {path}: {sorted(missing)}")
        grouped = {}
        current_scene_id = ""
        current_enabled = True
        for row in reader:
            scene_cell = str(row.get("scene_id", "")).strip()
            if scene_cell:
                current_scene_id = scene_cell
            if not current_scene_id:
                continue
            enabled_cell = str(row.get("enabled", "")).strip().lower()
            if enabled_cell:
                current_enabled = enabled_cell not in {"0", "false", "no"}
            bw = str(row["bw_mbps"]).strip()
            loss = str(row["loss_pct"]).strip()
            duration = str(row["duration_s"]).strip()
            if not bw or not loss or not duration:
                continue
            item = grouped.setdefault(
                current_scene_id,
                {"scene_id": current_scene_id, "enabled": current_enabled, "steps": []},
            )
            item["enabled"] = item["enabled"] and current_enabled
            item["steps"].append({"bw_mbps": bw, "loss_pct": loss, "duration_s": duration})
        return list(grouped.values())


def main():
    parser = argparse.ArgumentParser(description="Batch run scene captures from env.csv")
    parser.add_argument("--env-csv", default="env.csv", help="Scene table csv")
    parser.add_argument("--scenes", default=None, help="Comma-separated scene ids to run")
    parser.add_argument("--output-root", default="runs", help="Root output directory")
    parser.add_argument("--initial-stable-sleep", type=int, default=8)
    parser.add_argument("--post-scenario-tail-sleep", type=int, default=2)
    parser.add_argument("--pep-startup-sleep", type=float, default=2.0)
    parser.add_argument("--peer-ready-timeout", type=float, default=20.0)
    parser.add_argument("--iperf-max-time", type=int, default=3600)
    parser.add_argument("--decision-interval-ms", type=float, default=50.0)
    parser.add_argument("--sense-mode", default="rule", choices=["rule", "hybrid_gru"])
    parser.add_argument("--control-mode", default="legacy", choices=["legacy", "adaptive"])
    parser.add_argument("--hybrid-checkpoint", default=None)
    parser.add_argument("--hybrid-device", default="cpu")
    parser.add_argument("--sense-bw-ema-alpha", type=float, default=0.35)
    parser.add_argument("--sense-loss-ema-alpha", type=float, default=0.25)
    parser.add_argument("--adaptive-warmup-sec", type=float, default=5.0)
    parser.add_argument("--adaptive-min-bw-mbps", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    env_csv = Path(args.env_csv)
    if not env_csv.is_absolute():
        env_csv = (repo_root / env_csv).resolve()
    if not env_csv.exists():
        raise SystemExit(f"env csv not found: {env_csv}")

    requested = None
    if args.scenes:
        requested = {item.strip() for item in args.scenes.split(",") if item.strip()}

    scene_rows = load_scene_table(env_csv)
    selected = []
    for row in scene_rows:
        if not row["enabled"]:
            continue
        if requested is not None and row["scene_id"] not in requested:
            continue
        selected.append(row)

    if not selected:
        raise SystemExit("No scenes selected to run.")

    runner = (repo_root / "Mininet-scripts" / "run_single_scene_capture.py").resolve()
    started = time.time()

    for idx, row in enumerate(selected, start=1):
        scene_name = row["scene_id"]
        print(f"[batch] ({idx}/{len(selected)}) start {scene_name}")
        print(f"[batch] steps: {len(row['steps'])}")

        cmd = [
            sys.executable,
            str(runner),
            "--scene-name",
            scene_name,
            "--env-csv",
            str(env_csv),
            "--scene-id",
            scene_name,
            "--output-root",
            args.output_root,
            "--initial-stable-sleep",
            str(args.initial_stable_sleep),
            "--post-scenario-tail-sleep",
            str(args.post_scenario_tail_sleep),
            "--pep-startup-sleep",
            str(args.pep_startup_sleep),
            "--peer-ready-timeout",
            str(args.peer_ready_timeout),
            "--iperf-max-time",
            str(args.iperf_max_time),
            "--decision-interval-ms",
            str(args.decision_interval_ms),
            "--sense-mode",
            args.sense_mode,
            "--control-mode",
            args.control_mode,
            "--hybrid-device",
            args.hybrid_device,
            "--sense-bw-ema-alpha",
            str(args.sense_bw_ema_alpha),
            "--sense-loss-ema-alpha",
            str(args.sense_loss_ema_alpha),
            "--adaptive-warmup-sec",
            str(args.adaptive_warmup_sec),
            "--adaptive-min-bw-mbps",
            str(args.adaptive_min_bw_mbps),
        ]
        if args.hybrid_checkpoint:
            cmd.extend(["--hybrid-checkpoint", args.hybrid_checkpoint])
        if args.overwrite:
            cmd.append("--overwrite")

        scene_t0 = time.time()
        result = subprocess.run(cmd, cwd=str(repo_root))
        elapsed = time.time() - scene_t0
        if result.returncode != 0:
            raise SystemExit(f"[batch] scene failed: {scene_name}, exit={result.returncode}")
        print(f"[batch] done {scene_name} in {elapsed:.1f}s")

    print(f"[batch] all scenes finished in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
