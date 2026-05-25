#!/usr/bin/env bash
# Linux/WSL mirror of scripts/reproduce_all.ps1.
set -uo pipefail

REPORT_PATH="${1:-reports/phase7/reproduction.json}"
SKIP_TESTS="${SKIP_TESTS:-0}"

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
report_abs="${repo_root}/${REPORT_PATH}"
mkdir -p "$(dirname "${report_abs}")"

start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
started_epoch="$(date +%s)"

declare -a STAGE_KEYS=()
declare -A STAGE_STATUS=()
declare -A STAGE_DETAIL=()
declare -A STAGE_SECONDS=()
failed=0
warned=0

record() {
    local name="$1" status="$2" detail="$3" seconds="$4"
    STAGE_KEYS+=("$name")
    STAGE_STATUS["$name"]="$status"
    STAGE_DETAIL["$name"]="$detail"
    STAGE_SECONDS["$name"]="$seconds"
    printf "[%s] %s (%.1fs) -- %s\n" "$status" "$name" "$seconds" "$detail"
    case "${status}" in
        fail) failed=$((failed + 1)) ;;
        warn) warned=$((warned + 1)) ;;
    esac
}

invoke_stage() {
    local name="$1" warn_on_exit="${2:-}"
    shift 2
    local t0="$(date +%s.%N)"
    "$@"
    local rc=$?
    local t1="$(date +%s.%N)"
    local seconds
    seconds=$(awk "BEGIN { printf \"%.2f\", $t1 - $t0 }")
    if [[ "${rc}" -eq 0 ]]; then
        record "${name}" "ok" "exit=0" "${seconds}"
    elif [[ -n "${warn_on_exit}" && "${rc}" -eq "${warn_on_exit}" ]]; then
        record "${name}" "warn" "exit=${rc}" "${seconds}"
    else
        record "${name}" "fail" "exit=${rc}" "${seconds}"
    fi
}

cd "${repo_root}"

invoke_stage "verify_install" "1" bash scripts/verify_install.sh
invoke_stage "ruff_check" "" python -m ruff check src tests
invoke_stage "ruff_format_check" "" python -m ruff format --check src tests
invoke_stage "mypy_strict" "" python -m mypy src/hylog

if [[ "${SKIP_TESTS}" == "0" ]]; then
    invoke_stage "pytest" "" python -m pytest -q
else
    record "pytest" "warn" "skipped via SKIP_TESTS=1" "0.0"
fi

loso_out="reports/phase7/sample_loso"
rm -rf "${loso_out}"
invoke_stage "hylog_loso_mock" "" python -m hylog.cli.loso \
    --config configs/experiments/loso_hdfs_held.yaml \
    --out-dir "${loso_out}" \
    --mock --bootstrap-n 200

preds_path="${loso_out}/loso-hdfs-held-qwen25/hdfs/predictions.jsonl"
if [[ -f "${preds_path}" ]]; then
    cal_out="reports/phase7/sample_calibration"
    rm -rf "${cal_out}"
    invoke_stage "hylog_calibrate" "" python -m hylog.cli.calibrate \
        --predictions "${preds_path}" --out-dir "${cal_out}" --seed 42
else
    record "hylog_calibrate" "warn" "predictions.jsonl absent; skipped" "0.0"
fi

abl_out="reports/phase7/sample_ablation"
rm -rf "${abl_out}"
invoke_stage "hylog_ablation" "" python -m hylog.cli.ablation \
    --all-axes configs/ablation --out-dir "${abl_out}" --mock

elapsed=$(( $(date +%s) - started_epoch ))
finished_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

if [[ "${failed}" -gt 0 ]]; then
    verdict="BLOCK: ${failed} stage(s) failed"
    exit_code=2
elif [[ "${warned}" -gt 0 ]]; then
    verdict="PROCEED-WITH-CAVEATS: ${warned} warning(s)"
    exit_code=1
else
    verdict="PROCEED: every stage green"
    exit_code=0
fi

{
    echo "{"
    echo "  \"started_at_utc\": \"${start_utc}\","
    echo "  \"finished_at_utc\": \"${finished_utc}\","
    echo "  \"wallclock_seconds\": ${elapsed},"
    echo "  \"repo_root\": \"${repo_root}\","
    echo "  \"stages\": {"
    last=$((${#STAGE_KEYS[@]} - 1))
    for i in "${!STAGE_KEYS[@]}"; do
        k="${STAGE_KEYS[$i]}"
        comma=","
        [[ $i -eq $last ]] && comma=""
        det="${STAGE_DETAIL[$k]//\\/\\\\}"
        det="${det//\"/\\\"}"
        echo "    \"${k}\": { \"status\": \"${STAGE_STATUS[$k]}\", \"detail\": \"${det}\", \"seconds\": ${STAGE_SECONDS[$k]} }${comma}"
    done
    echo "  },"
    echo "  \"verdict\": \"${verdict}\","
    echo "  \"exit_code\": ${exit_code}"
    echo "}"
} > "${report_abs}"

echo ""
echo "Reproduce-all verdict: ${verdict}"
echo "Total wallclock: ${elapsed}s"
echo "Report: ${report_abs}"
exit "${exit_code}"
