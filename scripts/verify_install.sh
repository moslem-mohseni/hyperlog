#!/usr/bin/env bash
# Linux/WSL mirror of scripts/verify_install.ps1.
set -uo pipefail

REPORT_PATH="${1:-reports/phase7/verify_install.json}"
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
report_abs="${repo_root}/${REPORT_PATH}"
mkdir -p "$(dirname "${report_abs}")"

declare -a STEP_KEYS=()
declare -A STEP_STATUS=()
declare -A STEP_DETAIL=()
failed=0
warned=0

record() {
    local name="$1" status="$2" detail="$3"
    STEP_KEYS+=("$name")
    STEP_STATUS["$name"]="$status"
    STEP_DETAIL["$name"]="$detail"
    echo "[${status}] ${name} -- ${detail}"
    case "${status}" in
        fail) failed=$((failed + 1)) ;;
        warn) warned=$((warned + 1)) ;;
    esac
}

run_py() {
    python -c "$1" 2>/dev/null
}

if py_ver="$(python -c 'import sys; print(sys.version.split()[0])' 2>/dev/null)"; then
    record "python_version" "ok" "${py_ver}"
else
    record "python_version" "fail" "python not on PATH"
fi

if hy="$(run_py 'import hylog; print(hylog.__version__)')"; then
    record "import_hylog" "ok" "version=${hy}"
else
    record "import_hylog" "fail" "hylog not installed (run: pip install -e .)"
fi

for mod in torch transformers peft numpy scipy click yaml; do
    if ver="$(run_py "import ${mod}; print(getattr(${mod}, '__version__', 'unknown'))")"; then
        record "import_${mod}" "ok" "version=${ver}"
    else
        record "import_${mod}" "fail" "not importable"
    fi
done

for mod in bitsandbytes mlflow matplotlib; do
    if ver="$(run_py "import ${mod}; print(getattr(${mod}, '__version__', 'unknown'))")"; then
        record "import_${mod}" "ok" "version=${ver}"
    else
        record "import_${mod}" "warn" "optional dep not importable"
    fi
done

if cuda_av="$(run_py 'import torch; print(torch.cuda.is_available())')"; then
    if [[ "${cuda_av}" == "True" ]]; then
        dev=$(run_py 'import torch; print(torch.cuda.get_device_name(0))')
        record "cuda_available" "ok" "True device=${dev}"
        vram_mb=$(run_py 'import torch; print(torch.cuda.get_device_properties(0).total_memory // 1024 // 1024)')
        if [[ "${vram_mb}" =~ ^[0-9]+$ ]]; then
            if (( vram_mb >= 20000 )); then
                record "vram_gib" "ok" "${vram_mb} MiB (>= 20 GiB)"
            elif (( vram_mb >= 6000 )); then
                record "vram_gib" "warn" "${vram_mb} MiB -- enough for inference, NOT enough for full training"
            else
                record "vram_gib" "warn" "${vram_mb} MiB -- below inference floor"
            fi
        fi
    else
        record "cuda_available" "warn" "CUDA not available -- inference will be slow, training impossible"
    fi
fi

for cli in train predict loso calibrate ablation; do
    if out=$(run_py "from importlib import import_module; m = import_module('hylog.cli.${cli}'); print('ok' if hasattr(m, 'main') else 'no-main')"); then
        if [[ "${out}" == "ok" ]]; then
            record "cli_hylog-${cli}" "ok" "entry point present"
        else
            record "cli_hylog-${cli}" "warn" "no main()"
        fi
    else
        record "cli_hylog-${cli}" "warn" "entry point not importable"
    fi
done

if [[ "${failed}" -gt 0 ]]; then
    verdict="BLOCK: ${failed} step(s) failed -- environment unusable"
    exit_code=2
elif [[ "${warned}" -gt 0 ]]; then
    verdict="PROCEED-WITH-CAVEATS: ${warned} warning(s)"
    exit_code=1
else
    verdict="PROCEED: every check green"
    exit_code=0
fi

{
    echo "{"
    echo "  \"timestamp_utc\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
    echo "  \"repo_root\": \"${repo_root}\","
    echo "  \"steps\": {"
    last=$((${#STEP_KEYS[@]} - 1))
    for i in "${!STEP_KEYS[@]}"; do
        k="${STEP_KEYS[$i]}"
        comma=","
        [[ $i -eq $last ]] && comma=""
        det="${STEP_DETAIL[$k]//\\/\\\\}"
        det="${det//\"/\\\"}"
        echo "    \"${k}\": { \"status\": \"${STEP_STATUS[$k]}\", \"detail\": \"${det}\" }${comma}"
    done
    echo "  },"
    echo "  \"verdict\": \"${verdict}\","
    echo "  \"exit_code\": ${exit_code}"
    echo "}"
} > "${report_abs}"

echo ""
echo "Verify-install verdict: ${verdict}"
echo "Report: ${report_abs}"
exit "${exit_code}"
