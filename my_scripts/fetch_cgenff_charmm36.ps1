Param(
  [string]$DataDir = 'data',
  [string]$Charmm36Url = 'https://mackerell.umaryland.edu/download.php?filename=CHARMM_ff_params_files/charmm36-jul2021.ff.tgz',
  # 默认改为 Lemkul-Lab 的 cgenff_charmm2gmx 脚本（RAW 链接）
  [string]$CGenFFScriptUrl = 'https://raw.githubusercontent.com/Lemkul-Lab/cgenff_charmm2gmx/main/cgenff_charmm2gmx.py'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Ensure TLS 1.2 for GitHub/HTTPS
try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch {}

function Ensure-Dir([string]$p) { if (-not (Test-Path -LiteralPath $p)) { New-Item -ItemType Directory -Path $p | Out-Null } }

Ensure-Dir -p $DataDir
Push-Location -LiteralPath $DataDir

try {
  Write-Host "[1/3] Downloading CHARMM36 force field..." -ForegroundColor Cyan
  $ffTar = Split-Path -Leaf $Charmm36Url
  if (-not $ffTar -or -not ($ffTar -match '\.ff\.tgz$')) { $ffTar = 'charmm36.ff.tgz' }
  Write-Host "URL: $Charmm36Url" -ForegroundColor DarkGray
  try {
    Invoke-WebRequest -UseBasicParsing -Uri $Charmm36Url -OutFile $ffTar
    Write-Host "Saved: $(Resolve-Path -LiteralPath $ffTar)" -ForegroundColor Green
  } catch {
    Write-Host "Primary URL failed. Please manually download CHARMM36 and place the .ff folder and archive under $DataDir." -ForegroundColor Yellow
  }

  Write-Host "[2/3] Extracting: $ffTar" -ForegroundColor Cyan
  if (Test-Path -LiteralPath $ffTar) {
    if (Get-Command tar -ErrorAction SilentlyContinue) {
      tar -xzf $ffTar
    } else {
      Write-Host "No 'tar' found. Please install tar or extract $ffTar manually into $DataDir." -ForegroundColor Yellow
    }
  } else {
    Write-Host "Skipping extraction because archive not present." -ForegroundColor Yellow
  }

  Write-Host "[3/3] Downloading cgenff converter script..." -ForegroundColor Cyan
  $urls = @(
    $CGenFFScriptUrl,
    'https://raw.githubusercontent.com/Lemkul-Lab/cgenff_charmm2gmx/master/cgenff_charmm2gmx.py',
    'https://raw.githubusercontent.com/charmm-gui/cgenff/master/cgenff_charmm2gromacs.py',
    'https://raw.githubusercontent.com/charmm-gui/cgenff/main/cgenff_charmm2gromacs.py'
  )
  $downloaded = $false
  foreach ($u in $urls) {
    try {
      Write-Host "Trying: $u" -ForegroundColor DarkGray
      $leaf = Split-Path -Leaf $u
      if (-not $leaf.EndsWith('.py')) { $leaf = 'cgenff_charmm2gm_conv.py' }
      Invoke-WebRequest -UseBasicParsing -Uri $u -OutFile $leaf
      if ((Get-Item $leaf).Length -gt 0) { $downloaded = $true; $scriptName = $leaf; break }
    } catch {}
  }
  if ($downloaded) {
    Write-Host "Saved: $(Resolve-Path -LiteralPath $scriptName)" -ForegroundColor Green
  } else {
    Write-Host "Failed to download cgenff_charmm2gromacs.py. Please download manually and place into $DataDir." -ForegroundColor Yellow
  }

  # README
  @'
This folder contains third-party assets downloaded by scripts/fetch_cgenff_charmm36.ps1

- CHARMM36 for GROMACS (.ff directory), archive kept as .tgz
- cgenff_charmm2gromacs.py converter script

These are used by out/gmx_split_20250924_011827/make_topology_cgenff_auto.ps1
'@ | Set-Content -LiteralPath README.txt -Encoding UTF8

  Write-Host "Done. Assets are in: $(Resolve-Path .)" -ForegroundColor Green
}
finally {
  Pop-Location
}
