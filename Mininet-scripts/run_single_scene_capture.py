#!/usr/bin/env python3
import argparse
import csv
import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path

from mininet.link import TCLink
from mininet.log import info, setLogLevel
from mininet.net import Mininet
from mininet.node import Node
from mininet.topo import Topo

setLogLevel("info")

NODE_A_ETH1 = "10.0.0.1/24"
NODE_B_ETH1 = "10.0.0.2/24"
NODE_B_ETH2 = "10.0.1.2/24"
NODE_C_ETH2 = "10.0.1.3/24"
NODE_C_ETH1 = "10.0.2.3/24"
NODE_D_ETH1 = "10.0.2.4/24"


class LinuxRouter(Node):
    def config(self, **params):
        super().config(**params)
        self.cmd("sysctl net.ipv4.ip_forward=1")

    def terminate(self):
        self.cmd("sysctl net.ipv4.ip_forward=0")
        super().terminate()


class NetworkTopo(Topo):
    def build(self, **_opts):
        node_a = self.addNode("nodeA", cls=LinuxRouter, ip=NODE_A_ETH1)
        node_b = self.addNode("nodeB", cls=LinuxRouter, ip=NODE_B_ETH1)
        node_c = self.addNode("nodeC", cls=LinuxRouter, ip=NODE_C_ETH1)
        node_d = self.addNode("nodeD", cls=LinuxRouter, ip=NODE_D_ETH1)

        self.addLink(
            node_a,
            node_b,
            intfName1="nodeA-eth1",
            params1={"ip": NODE_A_ETH1},
            intfName2="nodeB-eth1",
            params2={"ip": NODE_B_ETH1},
        )
        self.addLink(
            node_c,
            node_d,
            intfName1="nodeC-eth1",
            params1={"ip": NODE_C_ETH1},
            intfName2="nodeD-eth1",
            params2={"ip": NODE_D_ETH1},
        )
        self.addLink(
            node_b,
            node_c,
            intfName1="nodeB-eth2",
            params1={"ip": NODE_B_ETH2},
            intfName2="nodeC-eth2",
            params2={"ip": NODE_C_ETH2},
            bw=20,
            delay="300ms",
            loss=1,
        )


def q(path: Path) -> str:
    return shlex.quote(str(path))


def mkdir_clean(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def start_bg(node, cwd: Path, command: str, logfile_path: Path):
    log_handle = open(logfile_path, "w", encoding="utf-8")
    proc = node.popen(
        ["bash", "-lc", command],
        cwd=str(cwd),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    return proc, log_handle


def stop_process(proc, grace_s: float = 2.0) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=max(grace_s, 0.0))
        return
    except subprocess.TimeoutExpired:
        pass
    proc.kill()
    try:
        proc.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        pass


def is_alive(proc) -> bool:
    return proc is not None and proc.poll() is None


def stop_iperf_client(node, server_ip: str, port: int, grace_s: float = 2.0) -> None:
    pattern = shlex.quote(f"iperf -c {server_ip} -p {port}")
    node.cmd(f"pkill -TERM -f {pattern} >/dev/null 2>&1 || true")
    deadline = time.time() + max(grace_s, 0.0)
    while time.time() < deadline:
        alive = node.cmd(f"pgrep -f {pattern} >/dev/null 2>&1; echo $?").strip().endswith("0")
        if not alive:
            return
        time.sleep(0.1)
    node.cmd(f"pkill -KILL -f {pattern} >/dev/null 2>&1 || true")


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copyfile(src, dst)


def log_contains(path: Path, needle: str) -> bool:
    if not path.exists():
        return False
    try:
        return needle in path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False


def wait_for_peer_ready(log_paths, timeout_s: float, poll_interval_s: float = 0.5) -> None:
    deadline = time.time() + timeout_s
    ready_phrase = "Connect peer PEPesc"
    while time.time() < deadline:
        if all(log_contains(path, ready_phrase) for path in log_paths):
            return
        time.sleep(poll_interval_s)
    pending = [str(path) for path in log_paths if not log_contains(path, ready_phrase)]
    raise TimeoutError(f"Timed out waiting for peer PEP readiness. Pending logs: {pending}")


def build_scene_config_from_env(env_csv: Path, scene_id: str, output_csv: Path) -> Path:
    rows = []
    current_scene_id = ""
    with open(env_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"scene_id", "bw_mbps", "loss_pct", "duration_s"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Missing columns in {env_csv}: {sorted(missing)}")
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
        raise SystemExit(f"Scene id not found or empty in {env_csv}: {scene_id}")

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["bw_mbps", "loss_pct", "duration_s"])
        writer.writeheader()
        writer.writerows(rows)
    return output_csv


def main():
    parser = argparse.ArgumentParser(description="Run one Mininet scene and capture PEPesc logs")
    parser.add_argument("--scene-name", required=True, help="Run directory name under output root")
    parser.add_argument(
        "--scenario-script",
        default="Mininet-scripts/scenario_from_csv.sh",
        help="Scenario shell script to run on nodeB",
    )
    parser.add_argument("--scenario-config", default=None, help="Scenario csv config file consumed by the scenario script")
    parser.add_argument("--env-csv", default=None, help="Unified env.csv that stores all scene steps")
    parser.add_argument("--scene-id", default=None, help="Scene id to extract from env.csv")
    parser.add_argument("--output-root", default="runs", help="Output root directory")
    parser.add_argument(
        "--iperf-max-time",
        type=int,
        default=3600,
        help="Upper bound passed to iperf client; actual run stops after scenario ends plus tail sleep",
    )
    parser.add_argument("--iperf-port", type=int, default=10000, help="iperf TCP port")
    parser.add_argument("--initial-stable-sleep", type=int, default=8, help="Warmup passed to scenario script")
    parser.add_argument(
        "--post-scenario-tail-sleep",
        type=int,
        default=2,
        help="Extra seconds to keep iperf running after the scenario script finishes",
    )
    parser.add_argument(
        "--pep-startup-sleep",
        type=float,
        default=2.0,
        help="Initial seconds to wait after both PEP endpoints start before readiness polling",
    )
    parser.add_argument(
        "--peer-ready-timeout",
        type=float,
        default=20.0,
        help="Timeout for waiting until both PEP logs report successful peer connection",
    )
    parser.add_argument("--pep-port", type=int, default=9999, help="PEPesc listen port")
    parser.add_argument("--decision-interval-ms", type=float, default=50.0, help="PEPesc sampling interval")
    parser.add_argument("--sense-mode", default="rule", choices=["rule", "hybrid_gru"], help="PEPesc online sensing mode")
    parser.add_argument("--control-mode", default="legacy", choices=["legacy", "adaptive"], help="PEPesc control mode")
    parser.add_argument("--hybrid-checkpoint", default=None, help="Hybrid GRU checkpoint path for pep.py")
    parser.add_argument("--hybrid-device", default="cpu", help="Hybrid runtime device for pep.py")
    parser.add_argument("--sense-bw-ema-alpha", type=float, default=0.35, help="EMA alpha for sensed bandwidth before DecisionController")
    parser.add_argument("--sense-loss-ema-alpha", type=float, default=0.25, help="EMA alpha for sensed loss rate before DecisionController")
    parser.add_argument("--adaptive-warmup-sec", type=float, default=5.0, help="Seconds after first ACK before adaptive takeover")
    parser.add_argument("--adaptive-min-bw-mbps", type=float, default=2.0, help="Minimum estimated bandwidth required before adaptive takeover")
    parser.add_argument(
        "--libstreamc-subdir",
        default="24.04",
        help="Subdirectory under libstreamc used to extend LD_LIBRARY_PATH",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory if it exists")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_root = (repo_root / args.output_root).resolve()
    run_dir = output_root / args.scene_name
    tmp_b_dir = run_dir / ".tmp_nodeB"
    tmp_c_dir = run_dir / ".tmp_nodeC"
    nodeb_pep_stdout = run_dir / "pep_b.stdout.log"
    nodec_pep_stdout = run_dir / "pep_c.stdout.log"

    if run_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Output directory already exists: {run_dir}. Use --overwrite to replace it.")
        shutil.rmtree(run_dir)

    mkdir_clean(run_dir)
    mkdir_clean(tmp_b_dir)
    mkdir_clean(tmp_c_dir)

    scenario_script = (repo_root / args.scenario_script).resolve()
    if not scenario_script.exists():
        raise SystemExit(f"Scenario script not found: {scenario_script}")
    scenario_config = None
    if args.scenario_config:
        scenario_config = Path(args.scenario_config)
        if not scenario_config.is_absolute():
            scenario_config = (repo_root / scenario_config).resolve()
        if not scenario_config.exists():
            raise SystemExit(f"Scenario config not found: {scenario_config}")
    else:
        if not args.env_csv or not args.scene_id:
            raise SystemExit("Either --scenario-config or (--env-csv and --scene-id) is required.")
        env_csv = Path(args.env_csv)
        if not env_csv.is_absolute():
            env_csv = (repo_root / env_csv).resolve()
        if not env_csv.exists():
            raise SystemExit(f"env csv not found: {env_csv}")
        scenario_config = build_scene_config_from_env(env_csv, args.scene_id, run_dir / "_resolved_scene_config.csv")

    deploy_b = (repo_root / "Mininet-scripts" / "deploy-proxy-on-node-b.sh").resolve()
    deploy_c = (repo_root / "Mininet-scripts" / "deploy-proxy-on-node-c.sh").resolve()
    remove_b = (repo_root / "Mininet-scripts" / "remove-proxy-on-node-b.sh").resolve()
    remove_c = (repo_root / "Mininet-scripts" / "remove-proxy-on-node-c.sh").resolve()
    pep_py = (repo_root / "pep.py").resolve()
    libstreamc_dir = (repo_root / "libstreamc" / args.libstreamc_subdir).resolve()

    net = Mininet(topo=NetworkTopo(), link=TCLink, controller=None)
    bg_procs = []
    started_at = time.time()
    node_a = None
    node_b = None
    node_c = None

    try:
        net.start()
        info("*** Topology started\n")

        node_a = net.getNodeByName("nodeA")
        node_b = net.getNodeByName("nodeB")
        node_c = net.getNodeByName("nodeC")
        node_d = net.getNodeByName("nodeD")

        node_a.cmd(f"route add default gw {NODE_B_ETH1.split('/')[0]}")
        node_b.cmd(f"route add default gw {NODE_C_ETH2.split('/')[0]}")
        node_c.cmd(f"route add default gw {NODE_B_ETH2.split('/')[0]}")
        node_d.cmd(f"route add default gw {NODE_C_ETH1.split('/')[0]}")

        node_b.cmd(f"bash {q(deploy_b)}")
        node_c.cmd(f"bash {q(deploy_c)}")

        pep_env = f"export LD_LIBRARY_PATH={shlex.quote(str(libstreamc_dir))}:$LD_LIBRARY_PATH"
        pep_common = (
            f"{pep_env} && "
            f"exec env PYTHONUNBUFFERED=1 python3 -u {q(pep_py)} "
            f"--senseMode {args.sense_mode} --controlMode {args.control_mode} "
            f"--decisionIntervalMs {args.decision_interval_ms} "
            f"--senseBwEmaAlpha {args.sense_bw_ema_alpha} "
            f"--senseLossEmaAlpha {args.sense_loss_ema_alpha} "
            f"--adaptiveWarmupSec {args.adaptive_warmup_sec} "
            f"--adaptiveMinBwMbps {args.adaptive_min_bw_mbps} "
            f"--hybridDevice {args.hybrid_device} --detail"
        )
        if args.hybrid_checkpoint:
            pep_common += f" --hybridCheckpoint {shlex.quote(str(Path(args.hybrid_checkpoint).resolve()))}"

        proc, log_handle = start_bg(
            node_b,
            tmp_b_dir,
            pep_common + f" --selfIp 10.0.1.2 --selfPort {args.pep_port} --peerIp 10.0.1.3 --peerPort {args.pep_port}",
            nodeb_pep_stdout,
        )
        bg_procs.append((proc, log_handle))
        proc, log_handle = start_bg(
            node_c,
            tmp_c_dir,
            pep_common + f" --selfIp 10.0.1.3 --selfPort {args.pep_port} --peerIp 10.0.1.2 --peerPort {args.pep_port}",
            nodec_pep_stdout,
        )
        bg_procs.append((proc, log_handle))

        proc, log_handle = start_bg(
            node_d,
            run_dir,
            f"exec iperf -s -p {args.iperf_port} -i 1",
            run_dir / "iperf_d.log",
        )
        bg_procs.append((proc, log_handle))

        time.sleep(max(args.pep_startup_sleep, 0.0))
        info("*** Waiting for B/C PEP peers to become ready\n")
        wait_for_peer_ready([nodeb_pep_stdout, nodec_pep_stdout], timeout_s=args.peer_ready_timeout)

        scenario_cmd = (
            f"exec env IFACE=nodeB-eth2 INITIAL_STABLE_SLEEP={args.initial_stable_sleep} "
            f"POST_SCENARIO_TAIL_SLEEP=0 "
            f"SCENARIO_CONFIG={shlex.quote(str(scenario_config))} "
            f"SCENARIO_LOG=scenario.csv bash {q(scenario_script)}"
        )
        scenario_proc, log_handle = start_bg(node_b, run_dir, scenario_cmd, run_dir / "scenario_runner.log")
        bg_procs.append((scenario_proc, log_handle))

        iperf_cmd = (
            f"exec iperf -c 10.0.2.4 -p {args.iperf_port} -i 1 -t {args.iperf_max_time}"
        )
        iperf_proc, log_handle = start_bg(node_a, run_dir, iperf_cmd, run_dir / "iperf_a.log")
        bg_procs.append((iperf_proc, log_handle))
        info("*** Running iperf client on nodeA until scenario completes\n")

        while is_alive(scenario_proc):
            time.sleep(0.5)

        time.sleep(max(args.post_scenario_tail_sleep, 0))
        stop_process(iperf_proc)
        stop_iperf_client(node_a, "10.0.2.4", args.iperf_port, grace_s=0.5)
        time.sleep(1.0)

    finally:
        for proc, _log_handle in reversed(bg_procs):
            stop_process(proc)

        try:
            if node_a is not None:
                stop_iperf_client(node_a, "10.0.2.4", args.iperf_port, grace_s=0.5)
        except Exception:
            pass

        try:
            if node_b is not None:
                node_b.cmd(f"bash {q(remove_b)}")
        except Exception:
            pass
        try:
            if node_c is not None:
                node_c.cmd(f"bash {q(remove_c)}")
        except Exception:
            pass

        net.stop()
        os.system("mn -c >/dev/null 2>&1")

        for _proc, log_handle in bg_procs:
            try:
                log_handle.close()
            except Exception:
                pass

    copy_if_exists(tmp_b_dir / "link_state.csv", run_dir / "link_state.csv")

    meta = {
        "scene_name": args.scene_name,
        "scenario_script": str(scenario_script),
        "scenario_config": str(scenario_config),
        "env_csv": str(args.env_csv) if args.env_csv else None,
        "scene_id": args.scene_id,
        "iperf_max_time": args.iperf_max_time,
        "iperf_port": args.iperf_port,
        "pep_port": args.pep_port,
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
        "finished_at": time.time(),
        "output_dir": str(run_dir),
    }
    with open(run_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    shutil.rmtree(tmp_b_dir, ignore_errors=True)
    shutil.rmtree(tmp_c_dir, ignore_errors=True)

    print(f"Single-scene capture finished: {run_dir}")
    print(f"Primary files: {run_dir / 'link_state.csv'} and {run_dir / 'scenario.csv'}")


if __name__ == "__main__":
    main()
