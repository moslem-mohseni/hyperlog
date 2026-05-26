#!/usr/bin/env bash
# Build the HyperLog paper PDF on Linux / WSL.
set -uo pipefail

PAPER_DIR="${1:-paper}"
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "${repo_root}"

echo "[1/2] Regenerating figures..."
if ! python scripts/build_figures.py --out-dir "${PAPER_DIR}/figures"; then
    echo "build_figures.py failed" >&2
    exit 1
fi

if ! command -v latexmk >/dev/null 2>&1; then
    echo ""
    echo "[!] latexmk not found on PATH."
    echo "    Install TeX Live ('sudo apt install texlive-latex-recommended"
    echo "    latexmk') and re-run: latexmk -pdf ${PAPER_DIR}/main.tex"
    exit 2
fi

echo "[2/2] Running latexmk..."
( cd "${PAPER_DIR}" && latexmk -pdf -interaction=nonstopmode -file-line-error main.tex )
rc=$?

if [[ "${rc}" -eq 0 ]]; then
    echo ""
    echo "Paper built: ${PAPER_DIR}/main.pdf"
fi
exit "${rc}"
