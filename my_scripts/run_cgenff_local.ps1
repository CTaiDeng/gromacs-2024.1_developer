Param(
  [Parameter(Mandatory=$true)][string]$CGenFFBin,
  [Parameter(Mandatory=$true)][string]$Mol2,
  [Parameter(Mandatory=$true)][string]$OutDir,
  [string]$CharmmFFDir,
  [string]$LigandCode,
  [string[]]$ExtraArgs
)

<#
离线生成 CGenFF 参数（.str）：
- 本脚本是对本地 cgenff 可执行文件的薄封装（需自行获取/安装）。
- 不同版本 cgenff 的命令行略有差异，请以 `& $CGenFFBin -h` 为准；
  本脚本尝试两种常见调用：
    1) cgenff -i ligand.mol2 -o ligand.str -f <charmm36.ff>
    2) cgenff ligand.mol2 ligand.str
- 生成成功后，输出到 $OutDir 下的 <基名>.str。

示例：
  pwsh -File scripts\run_cgenff_local.ps1 \ 
    -CGenFFBin "C:\\tools\\cgenff\\cgenff.exe" \ 
    -Mol2 out\gmx_split_20250924_011827\ligand.mol2 \ 
    -OutDir out\gmx_split_20250924_011827 \ 
    -CharmmFFDir scripts\data\charmm36-jul2021.ff
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Require-File([string]$p) { if (-not (Test-Path -LiteralPath $p)) { throw "Not found: $p" } }
function Ensure-Dir([string]$p) { if (-not (Test-Path -LiteralPath $p)) { New-Item -ItemType Directory -Path $p | Out-Null } }

Require-File $CGenFFBin
Require-File $Mol2
Ensure-Dir $OutDir

$mol2Path = (Resolve-Path -LiteralPath $Mol2).Path
$outDir = (Resolve-Path -LiteralPath $OutDir).Path
$stem = [System.IO.Path]::GetFileNameWithoutExtension($mol2Path)
$strOut = Join-Path $outDir ($stem + '.str')
$log = Join-Path $outDir ($stem + '.cgenff.log')

Write-Host "[CGenFF] Generating .str for: $mol2Path" -ForegroundColor Cyan

function Try-Run([string[]]$cmd) {
  Write-Host ("CMD> " + ($cmd -join ' ')) -ForegroundColor DarkGray
  $p = Start-Process -FilePath $cmd[0] -ArgumentList $cmd[1..($cmd.Length-1)] -NoNewWindow -PassThru -Wait -RedirectStandardOutput $log -RedirectStandardError $log
  return $p.ExitCode
}

$tried = @()
# Call pattern 1: cgenff -i in.mol2 -o out.str -f <ffdir> [extra]
$cmd1 = @($CGenFFBin,'-i',$mol2Path,'-o',$strOut)
if ($CharmmFFDir) { $cmd1 += @('-f',(Resolve-Path -LiteralPath $CharmmFFDir).Path) }
if ($ExtraArgs) { $cmd1 += $ExtraArgs }
$tried += ,$cmd1

# Call pattern 2: cgenff in.mol2 out.str [extra]
$cmd2 = @($CGenFFBin,$mol2Path,$strOut)
if ($ExtraArgs) { $cmd2 += $ExtraArgs }
$tried += ,$cmd2

$ok = $false
foreach ($cmd in $tried) {
  try {
    $code = Try-Run $cmd
    if ($code -eq 0 -and (Test-Path -LiteralPath $strOut) -and ((Get-Item $strOut).Length -gt 0)) {
      $ok = $true; break
    }
  } catch {}
}

if (-not $ok) {
  Write-Host "CGenFF 执行未成功。请检查日志：$log，并用 \\"& $CGenFFBin -h\\" 查看正确参数。" -ForegroundColor Red
  throw "cgenff failed"
}

Write-Host "[OK] Generated: $strOut" -ForegroundColor Green

