<#
.SYNOPSIS
    Download the public Loghub / Loghub-2.0 datasets used by HyLog.

.DESCRIPTION
    Fetches the HDFS, BGL, Thunderbird, and OpenStack archives from their
    upstream mirrors and verifies their SHA-256 checksums against
    data/checksums.txt. On first run the file is populated with the values
    observed during this download; on subsequent runs any mismatch is
    treated as a fatal error.

    HyLog does NOT redistribute raw datasets. License attribution is in
    data/LICENSES.txt (auto-generated from data/licenses.yaml).

.PARAMETER OutDir
    Directory under which to place the downloaded archives. Defaults to
    data/raw.

.PARAMETER Dataset
    Optional dataset filter. One of: hdfs, bgl, thunderbird, openstack, all.
    Default: all.

.EXAMPLE
    .\scripts\download_data.ps1 -Dataset hdfs
#>

[CmdletBinding()]
param(
    [string]$OutDir = "data/raw",
    [ValidateSet("hdfs", "bgl", "thunderbird", "openstack", "all")]
    [string]$Dataset = "all"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$checksumsPath = Join-Path $repoRoot "data\checksums.txt"
$outRoot = Join-Path $repoRoot $OutDir
New-Item -ItemType Directory -Path $outRoot -Force | Out-Null

# Upstream URLs. These point to logpai/loghub-2.0 which is the canonical
# source for re-annotated splits.
$urls = @{
    hdfs        = "https://github.com/logpai/loghub-2.0/raw/main/HDFS/HDFS_v2.zip"
    bgl         = "https://github.com/logpai/loghub-2.0/raw/main/BGL/BGL.zip"
    thunderbird = "https://github.com/logpai/loghub-2.0/raw/main/Thunderbird/Thunderbird.zip"
    openstack   = "https://zenodo.org/records/8196385/files/OpenStack.tar.gz"
}

function Get-FileSha256([string]$Path) {
    return (Get-FileHash -Algorithm SHA256 -Path $Path).Hash.ToLowerInvariant()
}

function Read-Checksums([string]$Path) {
    $map = @{}
    if (-not (Test-Path $Path)) { return $map }
    foreach ($line in Get-Content -Path $Path) {
        $stripped = $line.Trim()
        if (-not $stripped -or $stripped.StartsWith("#")) { continue }
        $parts = $stripped -split "\s+", 2
        if ($parts.Count -eq 2) {
            $map[$parts[1]] = $parts[0].ToLowerInvariant()
        }
    }
    return $map
}

function Write-Checksums([string]$Path, $Map) {
    $header = @"
# SHA-256 checksums for upstream dataset archives used by HyLog.
# Auto-managed by scripts/download_data.ps1. Do not edit by hand.
"@
    $lines = @($header, "")
    foreach ($key in ($Map.Keys | Sort-Object)) {
        $lines += "$($Map[$key])  $key"
    }
    $lines -join "`n" | Set-Content -Path $Path -Encoding utf8
}

$targets = if ($Dataset -eq "all") { $urls.Keys } else { @($Dataset) }
$known = Read-Checksums $checksumsPath

foreach ($name in $targets) {
    $url = $urls[$name]
    $fileName = Split-Path -Leaf $url
    $rel = "$name/$fileName"
    $dest = Join-Path $outRoot $rel
    $destDir = Split-Path -Parent $dest
    New-Item -ItemType Directory -Path $destDir -Force | Out-Null

    if (Test-Path $dest) {
        Write-Host "[skip] $rel already present"
    } else {
        Write-Host "[get ] $url"
        Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing
    }

    $sha = Get-FileSha256 $dest
    if ($known.ContainsKey($rel)) {
        if ($known[$rel] -ne $sha) {
            throw "Checksum mismatch for $rel`n  expected: $($known[$rel])`n  actual:   $sha"
        }
        Write-Host "[ok  ] $rel  $sha"
    } else {
        $known[$rel] = $sha
        Write-Host "[new ] $rel  $sha"
    }
}

Write-Checksums $checksumsPath $known
Write-Host ""
Write-Host "Checksums recorded in $checksumsPath"
