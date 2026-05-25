#!/usr/bin/env bash
# Download the public Loghub / Loghub-2.0 datasets used by HyLog.
# Linux mirror of scripts/download_data.ps1.
set -euo pipefail

OUT_DIR="${1:-data/raw}"
DATASET="${2:-all}"

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
checksums="${repo_root}/data/checksums.txt"
out_root="${repo_root}/${OUT_DIR}"
mkdir -p "${out_root}"

declare -A URLS=(
    [hdfs]="https://github.com/logpai/loghub-2.0/raw/main/HDFS/HDFS_v2.zip"
    [bgl]="https://github.com/logpai/loghub-2.0/raw/main/BGL/BGL.zip"
    [thunderbird]="https://github.com/logpai/loghub-2.0/raw/main/Thunderbird/Thunderbird.zip"
    [openstack]="https://zenodo.org/records/8196385/files/OpenStack.tar.gz"
)

if [[ "${DATASET}" == "all" ]]; then
    targets=("${!URLS[@]}")
else
    targets=("${DATASET}")
fi

declare -A known
if [[ -f "${checksums}" ]]; then
    while IFS= read -r line; do
        [[ -z "${line}" || "${line}" =~ ^# ]] && continue
        sha="${line%% *}"
        rel="${line##* }"
        known["${rel}"]="${sha}"
    done < "${checksums}"
fi

for name in "${targets[@]}"; do
    url="${URLS[$name]}"
    file="$(basename "${url}")"
    rel="${name}/${file}"
    dest="${out_root}/${rel}"
    mkdir -p "$(dirname "${dest}")"

    if [[ -f "${dest}" ]]; then
        echo "[skip] ${rel} already present"
    else
        echo "[get ] ${url}"
        curl -fsSL "${url}" -o "${dest}"
    fi

    sha="$(sha256sum "${dest}" | awk '{print $1}')"
    if [[ -n "${known[$rel]:-}" ]]; then
        if [[ "${known[$rel]}" != "${sha}" ]]; then
            echo "Checksum mismatch for ${rel}" >&2
            echo "  expected: ${known[$rel]}" >&2
            echo "  actual:   ${sha}" >&2
            exit 1
        fi
        echo "[ok  ] ${rel}  ${sha}"
    else
        known["${rel}"]="${sha}"
        echo "[new ] ${rel}  ${sha}"
    fi
done

{
    echo "# SHA-256 checksums for upstream dataset archives used by HyLog."
    echo "# Auto-managed by scripts/download_data.{ps1,sh}. Do not edit by hand."
    echo ""
    for key in $(printf '%s\n' "${!known[@]}" | sort); do
        echo "${known[$key]}  ${key}"
    done
} > "${checksums}"

echo ""
echo "Checksums recorded in ${checksums}"
