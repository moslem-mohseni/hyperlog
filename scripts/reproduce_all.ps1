<#
.SYNOPSIS
    Single-command reproduction of the HyLog Phase 0-6 deliverables on Windows.

.DESCRIPTION
    Executes, in order:
      1. verify_install.ps1                     -- environment probe
      2. ruff + ruff format --check + mypy      -- static quality gates
      3. pytest                                  -- unit + integration tests
      4. hylog-loso --mock                       -- LOSO smoke run (CPU)
      5. hylog-calibrate                         -- calibration on the LOSO output
      6. hylog-ablation --all-axes --mock        -- full ablation matrix

    Emits a structured report at reports/phase7/reproduction.json with
    the per-step verdict and an exit code:
      0 -- every gate green
      1 -- warnings (e.g. no GPU)
      2 -- blocking failures

.PARAMETER ReportPath
    Where to write the report. Defaults to reports/phase7/reproduction.json.

.PARAMETER SkipTests
    Skip pytest. Used by the GPU CI runner which has its own pytest stage.

.EXAMPLE
    .\scripts\reproduce_all.ps1
#>

[CmdletBinding()]
param(
    [string]$ReportPath = "reports/phase7/reproduction.json",
    [switch]$SkipTests
)

$ErrorActionPreference = "Continue"

$repoRoot = Split-Path -Parent $PSScriptRoot
$reportAbs = Join-Path $repoRoot $ReportPath
New-Item -ItemType Directory -Path (Split-Path $reportAbs) -Force | Out-Null

$startUtc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
$started = Get-Date
$stages = [ordered]@{}
$failed = 0
$warned = 0

function Record-Stage {
    param([string]$Name, [string]$Status, [string]$Detail, [double]$Seconds)
    $script:stages[$Name] = [ordered]@{
        status   = $Status
        detail   = $Detail
        seconds  = [math]::Round($Seconds, 2)
    }
    Write-Host ("[{0}] {1} ({2}s) -- {3}" -f $Status, $Name, [math]::Round($Seconds, 1), $Detail)
    if ($Status -eq "fail") { $script:failed++ }
    elseif ($Status -eq "warn") { $script:warned++ }
}

function Invoke-Stage {
    param([string]$Name, [scriptblock]$Body, [string]$WarnOnExit = "")
    $t = Get-Date
    $output = & $Body 2>&1
    $duration = (Get-Date) - $t
    $exit = $LASTEXITCODE
    if ($exit -eq 0) {
        Record-Stage $Name "ok" "exit=0" $duration.TotalSeconds
    } elseif ($WarnOnExit -ne "" -and $exit -eq [int]$WarnOnExit) {
        Record-Stage $Name "warn" ("exit={0}" -f $exit) $duration.TotalSeconds
    } else {
        Record-Stage $Name "fail" ("exit={0}" -f $exit) $duration.TotalSeconds
    }
    return $output
}

Push-Location $repoRoot
try {
    # ---- 1. Environment probe (warn-on-1 because GPU may be absent on a dev box) ----
    Invoke-Stage "verify_install" {
        & powershell -NoProfile -ExecutionPolicy Bypass -File "scripts/verify_install.ps1"
    } -WarnOnExit "1" | Out-Null

    # ---- 2. Static quality ----
    Invoke-Stage "ruff_check" { python -m ruff check src tests } | Out-Null
    Invoke-Stage "ruff_format_check" { python -m ruff format --check src tests } | Out-Null
    Invoke-Stage "mypy_strict" { python -m mypy src/hylog } | Out-Null

    # ---- 3. Pytest ----
    if (-not $SkipTests) {
        Invoke-Stage "pytest" { python -m pytest -q } | Out-Null
    } else {
        Record-Stage "pytest" "warn" "skipped via -SkipTests" 0.0
    }

    # ---- 4. LOSO smoke run (CPU mock) ----
    $losoOut = Join-Path $repoRoot "reports/phase7/sample_loso"
    if (Test-Path $losoOut) { Remove-Item -Recurse -Force $losoOut }
    Invoke-Stage "hylog_loso_mock" {
        python -m hylog.cli.loso `
            --config configs/experiments/loso_hdfs_held.yaml `
            --out-dir $losoOut `
            --mock --bootstrap-n 200
    } | Out-Null

    # ---- 5. Calibration on the LOSO output ----
    $predsPath = Join-Path $losoOut "loso-hdfs-held-qwen25/hdfs/predictions.jsonl"
    if (Test-Path $predsPath) {
        $calOut = Join-Path $repoRoot "reports/phase7/sample_calibration"
        if (Test-Path $calOut) { Remove-Item -Recurse -Force $calOut }
        Invoke-Stage "hylog_calibrate" {
            python -m hylog.cli.calibrate `
                --predictions $predsPath --out-dir $calOut --seed 42
        } | Out-Null
    } else {
        Record-Stage "hylog_calibrate" "warn" "predictions.jsonl absent; skipped" 0.0
    }

    # ---- 6. Ablation matrix ----
    $ablOut = Join-Path $repoRoot "reports/phase7/sample_ablation"
    if (Test-Path $ablOut) { Remove-Item -Recurse -Force $ablOut }
    Invoke-Stage "hylog_ablation" {
        python -m hylog.cli.ablation --all-axes configs/ablation --out-dir $ablOut --mock
    } | Out-Null

} finally {
    Pop-Location
}

$elapsed = ((Get-Date) - $started).TotalSeconds

if ($failed -gt 0) {
    $verdict = "BLOCK: $failed stage(s) failed"
    $exitCode = 2
} elseif ($warned -gt 0) {
    $verdict = "PROCEED-WITH-CAVEATS: $warned warning(s)"
    $exitCode = 1
} else {
    $verdict = "PROCEED: every stage green"
    $exitCode = 0
}

$report = [ordered]@{
    started_at_utc    = $startUtc
    finished_at_utc   = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    wallclock_seconds = [math]::Round($elapsed, 2)
    repo_root         = $repoRoot
    stages            = $stages
    verdict           = $verdict
    exit_code         = $exitCode
}

$json = $report | ConvertTo-Json -Depth 6
$json | Set-Content -Path $reportAbs -Encoding utf8

Write-Host ""
Write-Host ("Reproduce-all verdict: " + $verdict)
Write-Host ("Total wallclock: {0:N1}s" -f $elapsed)
Write-Host ("Report: " + $reportAbs)
exit $exitCode
