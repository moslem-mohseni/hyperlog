<#
.SYNOPSIS
    Build the HyperLog paper PDF.

.DESCRIPTION
    1. Regenerates every figure from the JSON artefacts under reports/.
    2. Runs latexmk to build paper/main.pdf.
    If latexmk is not on PATH, a clear error is printed and the script
    exits with code 2 — the figures are still rebuilt.

.PARAMETER PaperDir
    Defaults to paper/.

.EXAMPLE
    .\scripts\build_paper.ps1
#>

[CmdletBinding()]
param(
    [string]$PaperDir = "paper"
)

$ErrorActionPreference = "Continue"

$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot

# 1. Figures.
Write-Host "[1/2] Regenerating figures..."
& python scripts/build_figures.py --out-dir (Join-Path $PaperDir "figures")
if ($LASTEXITCODE -ne 0) {
    Pop-Location
    exit $LASTEXITCODE
}

# 2. LaTeX build.
$latexmk = Get-Command latexmk -ErrorAction SilentlyContinue
if ($null -eq $latexmk) {
    Write-Host ""
    Write-Host "[!] latexmk not found on PATH."
    Write-Host "    Install MiKTeX or TeX Live, then re-run: latexmk -pdf $PaperDir/main.tex"
    Pop-Location
    exit 2
}

Write-Host "[2/2] Running latexmk..."
Push-Location $PaperDir
& latexmk -pdf -interaction=nonstopmode -file-line-error main.tex
$rc = $LASTEXITCODE
Pop-Location
Pop-Location

if ($rc -eq 0) {
    Write-Host ""
    Write-Host "Paper built: $PaperDir/main.pdf"
} else {
    Write-Host "latexmk exited with code $rc"
}
exit $rc
