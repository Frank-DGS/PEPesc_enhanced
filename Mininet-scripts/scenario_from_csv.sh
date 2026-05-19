#!/bin/bash
set -euo pipefail

IFACE="${IFACE:-nodeB-eth2}"
CLASS_ID="${CLASS_ID:-5:1}"
NETEM_HANDLE="${NETEM_HANDLE:-10:}"
SCENARIO_LOG="${SCENARIO_LOG:-scenario.csv}"
SCENARIO_CONFIG="${SCENARIO_CONFIG:-}"
INITIAL_STABLE_SLEEP="${INITIAL_STABLE_SLEEP:-8}"
POST_SCENARIO_TAIL_SLEEP="${POST_SCENARIO_TAIL_SLEEP:-0}"

if [ -z "${SCENARIO_CONFIG}" ]; then
    echo "SCENARIO_CONFIG is required" >&2
    exit 1
fi

if [ ! -f "${SCENARIO_CONFIG}" ]; then
    echo "Scenario config not found: ${SCENARIO_CONFIG}" >&2
    exit 1
fi

echo "ts,true_bw,true_loss_rate,stage_duration_s" > "${SCENARIO_LOG}"

to_loss_decimal() {
    python3 - <<PY
s="${1}".strip().replace("%","")
print(float(s) / 100.0)
PY
}

apply_link() {
    local bw="$1"
    local loss_str="$2"
    tc class change dev "${IFACE}" classid "${CLASS_ID}" htb rate "${bw}"Mbit ceil "${bw}"Mbit burst 15Kb cburst 1600b
    tc qdisc change dev "${IFACE}" parent "${CLASS_ID}" handle "${NETEM_HANDLE}" netem delay 300ms loss "${loss_str}"
}

log_scenario() {
    local bw="$1"
    local loss_str="$2"
    local duration_s="$3"
    local ts
    local loss_dec
    ts="$(python3 -c 'import time; print(time.time())')"
    loss_dec="$(to_loss_decimal "${loss_str}")"
    echo "${ts},${bw},${loss_dec},${duration_s}" >> "${SCENARIO_LOG}"
    echo "[scenario] ts=${ts} bw=${bw}Mbps loss=${loss_str} duration=${duration_s}s"
}

readarray -t scenario_rows < <(python3 - "${SCENARIO_CONFIG}" <<'PY'
import csv
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    required = {"bw_mbps", "loss_pct", "duration_s"}
    missing = required.difference(reader.fieldnames or [])
    if missing:
        raise SystemExit(f"Missing columns in {path}: {sorted(missing)}")
    for row in reader:
        bw = str(row["bw_mbps"]).strip()
        loss = str(row["loss_pct"]).strip()
        duration = str(row["duration_s"]).strip()
        if not bw or not loss or not duration:
            continue
        print(f"{bw},{loss},{duration}")
PY
)

if [ "${#scenario_rows[@]}" -eq 0 ]; then
    echo "Scenario config has no valid rows: ${SCENARIO_CONFIG}" >&2
    exit 1
fi

IFS=',' read -r first_bw first_loss first_duration <<< "${scenario_rows[0]}"
apply_link "${first_bw}" "${first_loss}"
echo "[scenario] primed initial link to bw=${first_bw}Mbps loss=${first_loss} without logging"
echo "[scenario] initial stable sleep ${INITIAL_STABLE_SLEEP}s on primed first-stage link"
sleep "${INITIAL_STABLE_SLEEP}"

for row in "${scenario_rows[@]}"; do
    IFS=',' read -r bw loss duration <<< "${row}"
    apply_link "${bw}" "${loss}"
    log_scenario "${bw}" "${loss}" "${duration}"
    sleep "${duration}"
done

if [ "${POST_SCENARIO_TAIL_SLEEP}" -gt 0 ]; then
    echo "[scenario] tail sleep ${POST_SCENARIO_TAIL_SLEEP}s"
    sleep "${POST_SCENARIO_TAIL_SLEEP}"
fi

echo "[scenario] unified experiment finished"
