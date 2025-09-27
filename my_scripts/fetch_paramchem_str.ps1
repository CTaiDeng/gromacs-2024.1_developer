Param(
  [Parameter(Mandatory=$true)][string]$Mol2,
  [Parameter(Mandatory=$true)][string]$OutDir,
  [switch]$Headless
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Ensure-Python() {
  foreach ($cmd in @('python3','py','python')) {
    try { if (Get-Command $cmd -ErrorAction Stop) { return $cmd } } catch {}
  }
  throw 'Python not found. Please install Python 3.'
}

function Ensure-Package([string]$pkg) {
  try { python - << 'PY'
import importlib,sys
sys.exit(0 if importlib.util.find_spec(sys.argv[1]) else 1)
PY
  } catch {}
  if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing $pkg ..." -ForegroundColor Cyan
    pip install $pkg --upgrade
  }
}

$py = Ensure-Python
Ensure-Package 'selenium'
Ensure-Package 'webdriver-manager'

$args = @('my_scripts/fetch_paramchem_str.py','--mol2', $Mol2, '--out', $OutDir)
if ($Headless) { $args += '--headless' }

Write-Host ("CMD> $py " + ($args -join ' ')) -ForegroundColor DarkGray
& $py @args
