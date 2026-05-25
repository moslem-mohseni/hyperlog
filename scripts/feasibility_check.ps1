<#
.SYNOPSIS
    Phase 2A — Feasibility check for the upstream LogLLM repository on Windows.

.DESCRIPTION
    Clones https://github.com/guanwei49/LogLLM, probes the local environment,
    runs a py_compile syntax check over the upstream sources, and emits a
    structured report at reports/phase2/feasibility.json. The script is
    purely diagnostic and never modifies the upstream sources.

    If feasibility fails, the kill-switch documented in docs/ROADMAP.md
    section 11 is invoked: (a) WSL2 mirror, (b) LogFiT fallback.

.PARAMETER WorkDir
    Where to clone LogLLM. Defaults to third_party/LogLLM.

.PARAMETER ReportPath
    JSON report location. Defaults to reports/phase2/feasibility.json.

.EXAMPLE
    .\scripts\feasibility_check.ps1
#>

[CmdletBinding()]
param(
    [string]$WorkDir = "third_party/LogLLM",
    [string]$ReportPath = "reports/phase2/feasibility.json"
)

$ErrorActionPreference = "Continue"

$repoRoot = Split-Path -Parent $PSScriptRoot
$workAbs = Join-Path $repoRoot $WorkDir
$reportAbs = Join-Path $repoRoot $ReportPath
New-Item -ItemType Directory -Path (Split-Path $reportAbs) -Force | Out-Null

$steps = [ordered]@{}

function Record-Step {
    param([string]$Name, [string]$Status, [string]$Detail)
    $script:steps[$Name] = [ordered]@{ status = $Status; detail = $Detail }
    Write-Host ("[{0}] {1} -- {2}" -f $Status, $Name, $Detail)
}

function Run-Python {
    param([string]$Code)
    $out = & python -c $Code 2>$null
    return [pscustomobject]@{ ok = ($LASTEXITCODE -eq 0); out = ($out -join " ").Trim() }
}

# 1. Python version.
$py = Run-Python "import sys; print(sys.version.split()[0])"
if ($py.ok) {
    Record-Step "python_version" "ok" $py.out
} else {
    Record-Step "python_version" "fail" "python not found"
}

# 2. Library imports.
foreach ($mod in @("torch", "transformers", "peft", "bitsandbytes")) {
    $r = Run-Python "import $mod; print($mod.__version__)"
    if ($r.ok) {
        Record-Step ("import_" + $mod) "ok" ("version=" + $r.out)
    } else {
        Record-Step ("import_" + $mod) "warn" "not importable in this env"
    }
}

# 3. CUDA probe.
$cuda = Run-Python "import torch; print(torch.cuda.is_available())"
if ($cuda.ok -and $cuda.out -eq "True") {
    $dev = Run-Python "import torch; print(torch.cuda.get_device_name(0))"
    Record-Step "cuda_available" "ok" ("True device=" + $dev.out)
} elseif ($cuda.ok) {
    Record-Step "cuda_available" "warn" "torch present but CUDA not available (CPU-only env)"
} else {
    Record-Step "cuda_available" "warn" "torch not importable"
}

# 4. Clone LogLLM.
$cloneOk = $false
if (Test-Path (Join-Path $workAbs ".git")) {
    Push-Location $workAbs
    $head = (& git rev-parse HEAD 2>$null).Trim()
    Pop-Location
    Record-Step "clone_logllm" "ok" ("already cloned at HEAD=" + $head)
    $cloneOk = $true
} else {
    New-Item -ItemType Directory -Path (Split-Path $workAbs) -Force | Out-Null
    & git clone --depth=1 "https://github.com/guanwei49/LogLLM.git" $workAbs *> $null
    if ($LASTEXITCODE -eq 0) {
        Push-Location $workAbs
        $head = (& git rev-parse HEAD 2>$null).Trim()
        Pop-Location
        Record-Step "clone_logllm" "ok" ("cloned at HEAD=" + $head)
        $cloneOk = $true
    } else {
        Record-Step "clone_logllm" "fail" "git clone failed"
    }
}

# 5. Syntax check via py_compile.
if ($cloneOk) {
    $pyFiles = @(Get-ChildItem -Path $workAbs -Recurse -Filter *.py -ErrorAction SilentlyContinue)
    $count = $pyFiles.Count
    $bad = 0
    foreach ($f in $pyFiles) {
        & python -m py_compile $f.FullName 2>$null
        if ($LASTEXITCODE -ne 0) { $bad++ }
    }
    if ($bad -eq 0) {
        Record-Step "syntax_check" "ok" ("{0} files, all syntactically valid" -f $count)
    } else {
        Record-Step "syntax_check" "warn" ("{0} files; {1} syntax errors" -f $count, $bad)
    }
}

# 6. Windows-incompatibility scan.
if ($cloneOk) {
    $incompat = 0
    foreach ($f in Get-ChildItem -Path $workAbs -Recurse -Filter *.py) {
        $content = Get-Content -Raw $f.FullName -ErrorAction SilentlyContinue
        if ($content -match "/dev/null") { $incompat++; continue }
        if ($content -match "/tmp/") { $incompat++; continue }
    }
    if ($incompat -eq 0) {
        Record-Step "windows_compat" "ok" "no POSIX-only patterns detected"
    } else {
        Record-Step "windows_compat" "warn" ("POSIX-only patterns in {0} files" -f $incompat)
    }
}

# 7. Verdict.
$failed = @($steps.Values | Where-Object { $_.status -eq "fail" }).Count
$warned = @($steps.Values | Where-Object { $_.status -eq "warn" }).Count

if ($failed -gt 0) {
    $verdict = "BLOCK: $failed step(s) failed -- kill-switch (WSL2 / LogFiT fallback) recommended"
} elseif ($warned -gt 0) {
    $verdict = "PROCEED-WITH-CAVEATS: $warned warning(s) -- Phase 2B may begin; document any deviations"
} else {
    $verdict = "PROCEED: environment passes all feasibility checks"
}

$report = [ordered]@{
    timestamp_utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    repo_root     = $repoRoot
    upstream_url  = "https://github.com/guanwei49/LogLLM"
    work_dir      = $workAbs
    steps         = $steps
    verdict       = $verdict
}

$json = $report | ConvertTo-Json -Depth 6
$json | Set-Content -Path $reportAbs -Encoding utf8
Write-Host ""
Write-Host ("Feasibility verdict: " + $verdict)
Write-Host ("Report: " + $reportAbs)
