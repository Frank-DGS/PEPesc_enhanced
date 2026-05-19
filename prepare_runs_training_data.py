import argparse
import shutil
from pathlib import Path
from typing import List

from analyze_current_csv import run_analysis
from adaptive_schema import DEFAULT_RESAMPLE_MS


def parse_scene_ids(scene_ids_text: str) -> List[str]:
    items = [item.strip() for item in scene_ids_text.split(",") if item.strip()]
    if not items:
        raise ValueError("No valid scene ids parsed.")
    return items


def discover_scene_ids(runs_root: Path) -> List[str]:
    scene_dirs = [path for path in runs_root.iterdir() if path.is_dir() and path.name.isdigit()]
    return [path.name for path in sorted(scene_dirs, key=lambda p: int(p.name))]


def main():
    parser = argparse.ArgumentParser(description="Batch process runs/<scene_id> into training-ready csv files")
    parser.add_argument("--runs-root", default="runs", help="Directory containing per-scene run folders")
    parser.add_argument("--data-dir", default="data_v2_50ms", help="Output directory for training csv files")
    parser.add_argument("--scene-ids", default=None, help="Comma-separated scene ids to process; default: discover all numeric folders")
    parser.add_argument(
        "--sc-packet-size",
        type=int,
        default=1465,
        help="ScPacketSize in bytes; if protocol.py differs, change this value",
    )
    parser.add_argument("--resample-ms", type=int, default=DEFAULT_RESAMPLE_MS)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    runs_root = Path(args.runs_root)
    if not runs_root.is_absolute():
        runs_root = (repo_root / runs_root).resolve()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (repo_root / data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    if not runs_root.exists():
        raise SystemExit(f"Runs root not found: {runs_root}")

    if args.scene_ids:
        scene_ids = parse_scene_ids(args.scene_ids)
    else:
        scene_ids = discover_scene_ids(runs_root)

    if not scene_ids:
        raise SystemExit(f"No scene folders found under {runs_root}")

    print(f"Processing scenes: {', '.join(scene_ids)}")
    print(f"Runs root: {runs_root}")
    print(f"Training data dir: {data_dir}")

    for idx, scene_id in enumerate(scene_ids, start=1):
        scene_dir = runs_root / scene_id
        link_path = scene_dir / "link_state.csv"
        scenario_path = scene_dir / "scenario.csv"
        if not link_path.exists():
            raise SystemExit(f"[{scene_id}] missing link_state.csv: {link_path}")
        if not scenario_path.exists():
            raise SystemExit(f"[{scene_id}] missing scenario.csv: {scenario_path}")

        print(f"\n[{idx}/{len(scene_ids)}] scene {scene_id}")
        result = run_analysis(
            link_path=link_path,
            scenario_path=scenario_path,
            outdir=scene_dir,
            sc_packet_size=args.sc_packet_size,
            resample_ms=args.resample_ms,
        )
        dst_csv = data_dir / f"aligned_link_state_{scene_id}_{args.resample_ms}ms.csv"
        shutil.copyfile(result["resampled_csv_path"], dst_csv)
        print(
            f"  kept_rows={result['after_rows']}/{result['before_rows']} "
            f"copied_to={dst_csv}"
        )

    print("\nAll scene processing finished.")


if __name__ == "__main__":
    main()
