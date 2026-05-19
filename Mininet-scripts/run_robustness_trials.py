#!/usr/bin/env python3
import argparse
import csv
import json
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


IPERF_INTERVAL_RE = re.compile(
    r"\[\s*\d+\]\s+"
    r"(?P<start>\d+(?:\.\d+)?)\s*-\s*(?P<end>\d+(?:\.\d+)?)\s+sec\s+"
    r".+?\s+(?P<bw>\d+(?:\.\d+)?)\s+(?P<unit>[KMG])bits/sec",
    re.IGNORECASE,
)

FAILURE_PATTERNS = (
    "Have tried heartbeat",
    "Broken pipe",
    "Connection reset",
    "Traceback",
    "Turn off instantly",
)


def load_scene_steps(env_csv: Path, scene_id: str):
    rows = []
    current_scene_id = ""
    with open(env_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"scene_id", "bw_mbps", "loss_pct", "duration_s"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit("Missing columns in {}: {}".format(env_csv, sorted(missing)))
        for row in reader:
            scene_cell = str(row.get("scene_id", "")).strip()
            if scene_cell:
                current_scene_id = scene_cell
            if current_scene_id != scene_id:
                continue
            bw = str(row.get("bw_mbps", "")).strip()
            loss = str(row.get("loss_pct", "")).strip()
            duration = str(row.get("duration_s", "")).strip()
            if not bw or not loss or not duration:
                continue
            rows.append({"bw_mbps": bw, "loss_pct": loss, "duration_s": duration})
    if not rows:
        raise SystemExit("Scene {} not found in {}".format(scene_id, env_csv))
    return rows


def write_looped_scene_config(base_steps, loops: int, output_csv: Path):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["bw_mbps", "loss_pct", "duration_s"])
        writer.writeheader()
        for _ in range(max(loops, 1)):
            for row in base_steps:
                writer.writerow(row)


def parse_iperf_intervals(log_path: Path):
    rows = []
    if not log_path.exists():
        return rows
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        match = IPERF_INTERVAL_RE.search(line)
        if not match:
            continue
        start_s = float(match.group("start"))
        end_s = float(match.group("end"))
        bw = float(match.group("bw"))
        unit = match.group("unit").upper()
        if unit == "G":
            bw_mbps = bw * 1000.0
        elif unit == "M":
            bw_mbps = bw
        else:
            bw_mbps = bw / 1000.0
        rows.append({"start_s": start_s, "end_s": end_s, "throughput_mbps": bw_mbps})
    return rows


def read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def detect_failure(run_dir: Path, min_monitor_time_s: float, zero_mbps_threshold: float, zero_streak_needed: int):
    pep_b = read_text_if_exists(run_dir / "pep_b.stdout.log")
    pep_c = read_text_if_exists(run_dir / "pep_c.stdout.log")
    iperf_a = read_text_if_exists(run_dir / "iperf_a.log")
    combined = "\n".join([pep_b, pep_c, iperf_a])
    for pattern in FAILURE_PATTERNS:
        if pattern in combined:
            return pattern

    intervals = parse_iperf_intervals(run_dir / "iperf_d.log")
    if not intervals:
        return None
    if intervals[-1]["end_s"] < min_monitor_time_s:
        return None
    recent = intervals[-zero_streak_needed:]
    if len(recent) < zero_streak_needed:
        return None
    if all(item["throughput_mbps"] <= zero_mbps_threshold for item in recent):
        return "zero_goodput_streak"
    return None


def stop_capture_process(proc: subprocess.Popen):
    if proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.SIGINT)
    except Exception:
        pass
    try:
        proc.wait(timeout=20.0)
        return
    except subprocess.TimeoutExpired:
        pass
    proc.terminate()
    try:
        proc.wait(timeout=10.0)
        return
    except subprocess.TimeoutExpired:
        pass
    proc.kill()
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        pass


def planned_scenario_duration_s(base_steps, loops: int, initial_stable_sleep: int, tail_sleep: int):
    scenario_s = 0.0
    for row in base_steps:
        scenario_s += float(row["duration_s"])
    return float(initial_stable_sleep + loops * scenario_s + tail_sleep)


def last_iperf_end_s(run_dir: Path) -> float:
    intervals = parse_iperf_intervals(run_dir / "iperf_d.log")
    if not intervals:
        return 0.0
    return float(intervals[-1]["end_s"])


def flatten_nested_run_dir(trial_root: Path, scene_name: str):
    nested_run_dir = trial_root / scene_name
    if not nested_run_dir.exists():
        return trial_root
    for child in list(nested_run_dir.iterdir()):
        target = trial_root / child.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        shutil.move(str(child), str(target))
    shutil.rmtree(nested_run_dir, ignore_errors=True)
    return trial_root


def salvage_trial_files(trial_root: Path, scene_name: str, args, started_at: float, finished_at: float):
    run_dir = flatten_nested_run_dir(trial_root, scene_name)

    tmp_b = run_dir / ".tmp_nodeB"
    tmp_c = run_dir / ".tmp_nodeC"

    if not (run_dir / "link_state.csv").exists():
        for candidate in [tmp_b / "link_state.csv", tmp_c / "link_state.csv"]:
            if candidate.exists():
                shutil.copyfile(candidate, run_dir / "link_state.csv")
                break

    if not (run_dir / "meta.json").exists():
        meta = {
            "scene_name": scene_name,
            "scenario_config": str((Path(args.output_root) / "_generated_scene_config.csv")),
            "base_scene_id": args.scene_id,
            "loops": args.loops,
            "iperf_max_time": args.iperf_max_time,
            "decision_interval_ms": args.decision_interval_ms,
            "sense_mode": args.sense_mode,
            "control_mode": args.control_mode,
            "hybrid_checkpoint": args.hybrid_checkpoint,
            "hybrid_device": args.hybrid_device,
            "sense_bw_ema_alpha": args.sense_bw_ema_alpha,
            "sense_loss_ema_alpha": args.sense_loss_ema_alpha,
            "adaptive_warmup_sec": args.adaptive_warmup_sec,
            "adaptive_min_bw_mbps": args.adaptive_min_bw_mbps,
            "initial_stable_sleep": args.initial_stable_sleep,
            "post_scenario_tail_sleep": args.post_scenario_tail_sleep,
            "pep_startup_sleep": args.pep_startup_sleep,
            "peer_ready_timeout": args.peer_ready_timeout,
            "started_at": started_at,
            "finished_at": finished_at,
            "output_dir": str(run_dir),
        }
        with open(run_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    shutil.rmtree(tmp_b, ignore_errors=True)
    shutil.rmtree(tmp_c, ignore_errors=True)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Run long-duration robustness trials without changing existing capture scripts")
    parser.add_argument("--base-env-csv", default="env.csv", help="Base env csv with source scene definitions")
    parser.add_argument("--scene-id", default="23", help="Base scene id to repeat")
    parser.add_argument("--loops", type=int, default=3, help="How many times to repeat the base scene")
    parser.add_argument("--trials", type=int, default=20, help="How many trials to run")
    parser.add_argument("--scene-name", default=None, help="Generated long-scene name; default is scene<id>_loop<loops>")
    parser.add_argument("--output-root", required=True, help="Root output directory for this method")
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
    parser.add_argument("--failure-zero-threshold-mbps", type=float, default=0.05)
    parser.add_argument("--failure-zero-streak-s", type=int, default=8)
    parser.add_argument("--failure-monitor-start-s", type=float, default=20.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    env_csv = Path(args.base_env_csv)
    if not env_csv.is_absolute():
        env_csv = (repo_root / env_csv).resolve()
    if not env_csv.exists():
        raise SystemExit("env csv not found: {}".format(env_csv))

    scene_name = args.scene_name or "scene{}_loop{}".format(args.scene_id, args.loops)
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (repo_root / output_root).resolve()

    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    base_steps = load_scene_steps(env_csv, args.scene_id)
    generated_config = output_root / "_generated_scene_config.csv"
    write_looped_scene_config(base_steps, args.loops, generated_config)
    planned_time_s = planned_scenario_duration_s(base_steps, args.loops, args.initial_stable_sleep, args.post_scenario_tail_sleep)

    runner = (repo_root / "Mininet-scripts" / "run_single_scene_capture.py").resolve()
    summary_rows = []

    for trial_id in range(1, args.trials + 1):
        trial_root = output_root / "trial_{}".format(trial_id)
        trial_root.mkdir(parents=True, exist_ok=True)
        nested_run_dir = trial_root / scene_name
        if nested_run_dir.exists():
            shutil.rmtree(nested_run_dir)

        cmd = [
            sys.executable,
            str(runner),
            "--scene-name", scene_name,
            "--scenario-config", str(generated_config),
            "--output-root", str(trial_root),
            "--iperf-max-time", str(args.iperf_max_time),
            "--initial-stable-sleep", str(args.initial_stable_sleep),
            "--post-scenario-tail-sleep", str(args.post_scenario_tail_sleep),
            "--pep-startup-sleep", str(args.pep_startup_sleep),
            "--peer-ready-timeout", str(args.peer_ready_timeout),
            "--decision-interval-ms", str(args.decision_interval_ms),
            "--sense-mode", args.sense_mode,
            "--control-mode", args.control_mode,
            "--hybrid-device", args.hybrid_device,
            "--sense-bw-ema-alpha", str(args.sense_bw_ema_alpha),
            "--sense-loss-ema-alpha", str(args.sense_loss_ema_alpha),
            "--adaptive-warmup-sec", str(args.adaptive_warmup_sec),
            "--adaptive-min-bw-mbps", str(args.adaptive_min_bw_mbps),
            "--overwrite",
        ]
        if args.hybrid_checkpoint:
            cmd.extend(["--hybrid-checkpoint", args.hybrid_checkpoint])

        print("[robustness] trial {}/{} start".format(trial_id, args.trials))
        t0 = time.time()
        proc = subprocess.Popen(cmd, cwd=str(repo_root))
        abort_reason = None

        while proc.poll() is None:
            time.sleep(1.0)
            run_dir = nested_run_dir if nested_run_dir.exists() else trial_root
            abort_reason = detect_failure(
                run_dir,
                min_monitor_time_s=args.failure_monitor_start_s,
                zero_mbps_threshold=args.failure_zero_threshold_mbps,
                zero_streak_needed=args.failure_zero_streak_s,
            )
            if abort_reason:
                print("[robustness] trial {} abort early: {}".format(trial_id, abort_reason))
                stop_capture_process(proc)
                break

        returncode = proc.wait()
        wall_time_s = time.time() - t0
        run_dir = salvage_trial_files(trial_root, scene_name, args, t0, time.time())
        completion_time_s = last_iperf_end_s(run_dir)
        completed = completion_time_s >= max(planned_time_s - 2.0, 0.0)
        status = "completed" if completed and returncode == 0 and not abort_reason else "aborted"

        row = {
            "trial_id": trial_id,
            "scene_name": scene_name,
            "base_scene_id": args.scene_id,
            "loops": args.loops,
            "status": status,
            "abort_reason": abort_reason or "",
            "returncode": returncode,
            "planned_time_s": planned_time_s,
            "completion_time_s": completion_time_s,
            "completion_ratio": (completion_time_s / planned_time_s) if planned_time_s > 0 else 0.0,
            "wall_time_s": wall_time_s,
            "run_dir": str(run_dir),
        }
        summary_rows.append(row)
        with open(trial_root / "trial_result.json", "w", encoding="utf-8") as f:
            json.dump(row, f, indent=2, ensure_ascii=False)
        print("[robustness] trial {} done status={} completion={:.1f}/{:.1f}s".format(trial_id, status, completion_time_s, planned_time_s))

    summary_path = output_root / "trial_summary.csv"
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "trial_id", "scene_name", "base_scene_id", "loops", "status", "abort_reason", "returncode",
                "planned_time_s", "completion_time_s", "completion_ratio", "wall_time_s", "run_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print("[robustness] summary written to {}".format(summary_path))


if __name__ == "__main__":
    main()
