Param(
  [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Ensure-Python() {
  foreach ($cmd in @('python3','py','python')) {
    try { if (Get-Command $cmd -ErrorAction Stop) { return $cmd } } catch {}
  }
  throw 'Python not found. Please install Python 3.'
}

$py = Ensure-Python
$script = Join-Path $PSScriptRoot 'align_my_documents.py'
$args = @($script)
if ($DryRun) { $args += @('--dry-run') }
Write-Host ("CMD> $py " + ($args -join ' ')) -ForegroundColor DarkGray
& $py @args
