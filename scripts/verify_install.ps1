<#
.SYNOPSIS
    Verify the HyLog environment on a fresh Windows machine.

.DESCRIPTION
    Probes Python version, key library imports, CUDA / GPU availability,
    bitsandbytes, and the editable install. Emits a structured JSON
    report at reports/phase7/verify_install.json and an exit code that
    CI can gate on (0 = green, 1 = warnings, 2 = blocking failure).

.PARAMETER ReportPath
    Where to write the report. Defaults to reports/phase7/verify_install.json.

.EXAMPLE
    .\scripts\verify_install.ps1
#>

[CmdletBinding()]
param(
    [string]$ReportPath = "reports/phase7/verify_install.json"
)

$ErrorActionPreference = "Continue"

$repoRoot = Split-Path -Parent $PSScriptRoot
$reportAbs = Join-Path $repoRoot $ReportPath
New-Item -ItemType Directory -Path (Split-Path $reportAbs) -Force | Out-Null

$steps = [ordered]@{}
$failed = 0
$warned = 0

function Record-Step {
    param([string]$Name, [string]$Status, [string]$Detail)
    $script:steps[$Name] = [ordered]@{ status = $Status; detail = $Detail }
    Write-Host ("[{0}] {1} -- {2}" -f $Status, $Name, $Detail)
    if ($Status -eq "fail") { $script:failed++ }
    elseif ($Status -eq "warn") { $script:warned++ }
}

function Run-Python {
    param([string]$Code)
    $out = & python -c $Code 2>$null
    return [pscustomobject]@{ ok = ($LASTEXITCODE -eq 0); out = ($out -join " ").Trim() }
}

# 1. Python.
$py = Run-Python "import sys; print(sys.version.split()[0])"
if ($py.ok) {
    Record-Step "python_version" "ok" $py.out
} else {
    Record-Step "python_version" "fail" "python not on PATH"
}

# 2. HyLog package import (editable install).
$hy = Run-Python "import hylog; print(hylog.__version__)"
if ($hy.ok) {
    Record-Step "import_hylog" "ok" ("version=" + $hy.out)
} else {
    Record-Step "import_hylog" "fail" "hylog not installed (run: pip install -e .)"
}

# 3. Core ML imports.
foreach ($mod in @("torch", "transformers", "peft", "numpy", "scipy", "click", "yaml")) {
    $r = Run-Python "import $mod; print(getattr($mod, '__version__', 'unknown'))"
    if ($r.ok) {
        Record-Step ("import_" + $mod) "ok" ("version=" + $r.out)
    } else {
        Record-Step ("import_" + $mod) "fail" "not importable"
    }
}

# 4. Optional imports.
foreach ($mod in @("bitsandbytes", "mlflow", "matplotlib")) {
    $r = Run-Python "import $mod; print(getattr($mod, '__version__', 'unknown'))"
    if ($r.ok) {
        Record-Step ("import_" + $mod) "ok" ("version=" + $r.out)
    } else {
        Record-Step ("import_" + $mod) "warn" "optional dep not importable"
    }
}

# 5. CUDA probe.
$cuda = Run-Python "import torch; print(torch.cuda.is_available())"
if ($cuda.ok -and $cuda.out -eq "True") {
    $dev = Run-Python "import torch; print(torch.cuda.get_device_name(0))"
    Record-Step "cuda_available" "ok" ("True device=" + $dev.out)
    $vram = Run-Python "import torch; print(torch.cuda.get_device_properties(0).total_memory // 1024 // 1024)"
    if ($vram.ok) {
        $vramMb = [int]$vram.out
        if ($vramMb -ge 20000) {
            Record-Step "vram_gib" "ok" ("{0:N0} MiB (>= 20 GiB)" -f $vramMb)
        } elseif ($vramMb -ge 6000) {
            Record-Step "vram_gib" "warn" ("{0:N0} MiB -- enough for inference, NOT enough for full training" -f $vramMb)
        } else {
            Record-Step "vram_gib" "warn" ("{0:N0} MiB -- below inference floor" -f $vramMb)
        }
    }
} else {
    Record-Step "cuda_available" "warn" "CUDA not available -- inference will be slow, training impossible"
}

# 6. CLI entry points.
foreach ($cli in @("hylog-train", "hylog-predict", "hylog-loso", "hylog-calibrate", "hylog-ablation")) {
    $r = Run-Python "from importlib import import_module; m = import_module('hylog.cli.' + '$cli'.split('-',1)[1].replace('-','_')); print('ok' if hasattr(m, 'main') else 'no-main')"
    if ($r.ok -and $r.out -eq "ok") {
        Record-Step ("cli_" + $cli) "ok" "entry point present"
    } else {
        Record-Step ("cli_" + $cli) "warn" "entry point not importable"
    }
}

# 7. Verdict.
if ($failed -gt 0) {
    $verdict = "BLOCK: $failed step(s) failed -- environment unusable"
    $exitCode = 2
} elseif ($warned -gt 0) {
    $verdict = "PROCEED-WITH-CAVEATS: $warned warning(s)"
    $exitCode = 1
} else {
    $verdict = "PROCEED: every check green"
    $exitCode = 0
}

$report = [ordered]@{
    timestamp_utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    repo_root     = $repoRoot
    steps         = $steps
    verdict       = $verdict
    exit_code     = $exitCode
}

$json = $report | ConvertTo-Json -Depth 6
$json | Set-Content -Path $reportAbs -Encoding utf8

Write-Host ""
Write-Host ("Verify-install verdict: " + $verdict)
Write-Host ("Report: " + $reportAbs)
exit $exitCode
