#!/usr/bin/env bash
# Phase 2A — Feasibility check for the upstream LogLLM repository.
# Linux/WSL mirror of scripts/feasibility_check.ps1.
set -euo pipefail

WORK_DIR="${1:-third_party/LogLLM}"
REPORT_PATH="${2:-reports/phase2/feasibility.json}"

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
work_abs="${repo_root}/${WORK_DIR}"
report_abs="${repo_root}/${REPORT_PATH}"
mkdir -p "$(dirname "${report_abs}")"

declare -a STEP_KEYS
declare -A STEP_STATUS
declare -A STEP_DETAIL

record() {
    local name="$1" status="$2" detail="$3"
    STEP_KEYS+=("$name")
    STEP_STATUS["$name"]="$status"
    STEP_DETAIL["$name"]="$detail"
    echo "[${status}] ${name} — ${detail}"
}

if py_ver="$(python --version 2>&1)"; then
    record "python_version" "ok" "${py_ver}"
else
    record "python_version" "fail" "python not found"
fi

for mod in torch transformers peft bitsandbytes; do
    if ver=$(python -c "import ${mod}; print(${mod}.__version__)" 2>&1); then
        record "import_${mod}" "ok" "version=${ver}"
    else
        record "import_${mod}" "warn" "${ver}"
    fi
done

if cuda_av=$(python -c "import torch; print(torch.cuda.is_available())" 2>&1); then
    name="no-cuda"
    if [[ "${cuda_av}" == "True" ]]; then
        name=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    fi
    record "cuda_available" "ok" "available=${cuda_av} device=${name}"
else
    record "cuda_available" "warn" "${cuda_av}"
fi

if [[ -d "${work_abs}/.git" ]]; then
    head=$(cd "${work_abs}" && git rev-parse HEAD)
    record "clone_logllm" "ok" "already cloned at HEAD=${head}"
else
    mkdir -p "$(dirname "${work_abs}")"
    if git clone --depth=1 https://github.com/guanwei49/LogLLM.git "${work_abs}" > /dev/null 2>&1; then
        head=$(cd "${work_abs}" && git rev-parse HEAD)
        record "clone_logllm" "ok" "cloned at HEAD=${head}"
    else
        record "clone_logllm" "fail" "git clone failed"
    fi
fi

if [[ -d "${work_abs}" ]]; then
    n_files=$(find "${work_abs}" -name "*.py" | wc -l | tr -d ' ')
    bad=0
    while IFS= read -r f; do
        if ! python -m py_compile "${f}" 2>/dev/null; then
            bad=$((bad + 1))
        fi
    done < <(find "${work_abs}" -name "*.py")
    if [[ "${bad}" -eq 0 ]]; then
        record "syntax_check" "ok" "${n_files} files, all syntactically valid"
    else
        record "syntax_check" "warn" "${n_files} files; ${bad} syntax errors"
    fi
fi

incompat=()
if [[ -d "${work_abs}" ]]; then
    while IFS= read -r f; do
        if grep -lq "/dev/null\|/tmp/" "${f}" 2>/dev/null; then
            incompat+=("${f}")
        fi
    done < <(find "${work_abs}" -name "*.py")
fi
if [[ ${#incompat[@]} -eq 0 ]]; then
    record "windows_compat" "ok" "no POSIX-only patterns detected"
else
    record "windows_compat" "warn" "POSIX-only patterns in ${#incompat[@]} files"
fi

failed=0
warned=0
for k in "${STEP_KEYS[@]}"; do
    case "${STEP_STATUS[$k]}" in
        fail) failed=$((failed + 1)) ;;
        warn) warned=$((warned + 1)) ;;
    esac
done

if [[ "${failed}" -gt 0 ]]; then
    verdict="BLOCK: ${failed} step(s) failed — kill-switch recommended"
elif [[ "${warned}" -gt 0 ]]; then
    verdict="PROCEED-WITH-CAVEATS: ${warned} warning(s)"
else
    verdict="PROCEED: environment passes all feasibility checks"
fi

{
    echo "{"
    echo "  \"timestamp_utc\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
    echo "  \"repo_root\": \"${repo_root}\","
    echo "  \"upstream_url\": \"https://github.com/guanwei49/LogLLM\","
    echo "  \"work_dir\": \"${work_abs}\","
    echo "  \"steps\": {"
    last_idx=$((${#STEP_KEYS[@]} - 1))
    for i in "${!STEP_KEYS[@]}"; do
        k="${STEP_KEYS[$i]}"
        comma=","
        [[ $i -eq $last_idx ]] && comma=""
        echo "    \"${k}\": { \"status\": \"${STEP_STATUS[$k]}\", \"detail\": \"${STEP_DETAIL[$k]//\"/\\\"}\" }${comma}"
    done
    echo "  },"
    echo "  \"verdict\": \"${verdict}\""
    echo "}"
} > "${report_abs}"

echo ""
echo "Feasibility verdict: ${verdict}"
echo "Report: ${report_abs}"
