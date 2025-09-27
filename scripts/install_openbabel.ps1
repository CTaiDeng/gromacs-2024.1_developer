Param(
  [string]$EnvName = 'base',
  [switch]$CreateEnv,
  [switch]$UseMamba,
  [string]$Channel = 'conda-forge'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Have-Cmd([string]$name) {
  try { return $null -ne (Get-Command $name -ErrorAction Stop) } catch { return $false }
}

function Run-Cmd([string]$exe, [string[]]$args) {
  Write-Host ("CMD> " + $exe + ' ' + ($args -join ' ')) -ForegroundColor DarkGray
  $p = Start-Process -FilePath $exe -ArgumentList $args -NoNewWindow -PassThru -Wait
  if ($p.ExitCode -ne 0) { throw "Command failed with exit code $($p.ExitCode): $exe" }
}

function Conda-Env-Exists([string]$name) {
  try {
    $out = conda env list | Out-String
    return ($out -split "`n") -match "^$name\s"
  } catch { return $false }
}

if (-not (Have-Cmd 'conda')) {
  Write-Host "未检测到 conda 命令。请在已激活的 Conda/Anaconda/Miniconda 终端中运行本脚本。" -ForegroundColor Red
  throw "conda not found"
}

$solver = 'conda'
if ($UseMamba -and (Have-Cmd 'mamba')) { $solver = 'mamba' }

if ($EnvName -ne 'base' -and $CreateEnv -and -not (Conda-Env-Exists $EnvName)) {
  Write-Host "创建 Conda 环境: $EnvName" -ForegroundColor Cyan
  Run-Cmd $solver @('create','-y','-n',$EnvName,'-c',$Channel,'python=3.11')
}

Write-Host "在环境 [$EnvName] 安装 Open Babel (openbabel) from $Channel" -ForegroundColor Cyan
Run-Cmd $solver @('install','-y','-n',$EnvName,'-c',$Channel,'openbabel')

Write-Host "验证 obabel 可用性..." -ForegroundColor Cyan
try {
  # 使用 conda run 以确保在指定环境中调用
  Run-Cmd 'conda' @('run','-n',$EnvName,'obabel','-V')
  Write-Host "Open Babel 安装并可用。" -ForegroundColor Green
} catch {
  Write-Host "验证失败。请尝试在该 Conda 环境中手动运行：`n  conda activate $EnvName; obabel -V" -ForegroundColor Yellow
}

Write-Host "提示：可用 Open Babel 将 PDB 转 mol2 示例：`n  conda run -n $EnvName obabel input.pdb -O output.mol2 --addh --gen3d" -ForegroundColor DarkGray

