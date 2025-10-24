# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

<#
安全的 UTF-8+LF 批量转换脚本。
- 统一将文本/源码文件重写为 UTF-8（无 BOM）+ LF 行尾。
- 默认 DryRun 预览，需 -Force 才实际落盘；保留 .utf8lf.bak 备份（可 -NoBackup 关闭）。
- 跳过常见二进制与大目录（.git、node_modules、out、.venv、kernel_reference）。

示例：
  # 预览将被转换的文件
  pwsh -File my_scripts/convert_to_utf8_lf.ps1 -DryRun

  # 实际执行（保留备份）
  pwsh -File my_scripts/convert_to_utf8_lf.ps1 -Force

  # 指定目录/扩展名，不保留备份
  pwsh -File my_scripts/convert_to_utf8_lf.ps1 -Path . -Extensions .py,.md,.json -Force -NoBackup
#>

param(
  [string]$Path = '.',
  [string[]]$Extensions = @(
    '.md','.txt','.json','.yml','.yaml','.toml','.ini','.cfg','.ps1','.psm1','.psd1',
    '.py','.sh','.bat','.cmd','.c','.h','.hpp','.cc','.cpp','.cu','.cl','.cmake','.ts','.tsx','.js','.jsx','.css','.scss','.html'
  ),
  [switch]$DryRun,
  [switch]$Force,
  [switch]$NoBackup
)

$ErrorActionPreference = 'Stop'

# 目录排除（正则）
$excludeDirs = @(
  '\\.git(\\|$)',
  'node_modules(\\|$)',
  'out(\\|$)',
  '(\\|^)\.venv(\\|$)',
  '(\\|^)venv(\\|$)',
  'my_docs\\project_docs\\kernel_reference(\\|$)'
)

function Test-IsTextFile([byte[]]$bytes){
  if ($bytes.Length -eq 0) { return $true }
  if ($bytes -contains 0) { return $false } # NUL 视为二进制
  $hi = ($bytes | Where-Object { $_ -gt 127 }).Count
  return ($hi / [double]$bytes.Length) -lt 0.3
}

function Write-Utf8NoBom([string]$filePath, [string]$text){
  $enc = New-Object System.Text.UTF8Encoding($false)
  [System.IO.File]::WriteAllText($filePath, $text, $enc)
}

$root = Resolve-Path -LiteralPath $Path
$all = Get-ChildItem -LiteralPath $root -Recurse -File -ErrorAction SilentlyContinue
$targets = @()
foreach($f in $all){
  $p = $f.FullName
  if ($excludeDirs | ForEach-Object { if ($p -match $_) { $true } }) { continue }
  if ($Extensions.Count -gt 0 -and -not ($Extensions -contains $f.Extension)) { continue }
  try {
    $bytes = [System.IO.File]::ReadAllBytes($p)
  } catch { continue }
  if (-not (Test-IsTextFile $bytes)) { continue }
  $targets += $f
}

$changed = 0; $errors = 0; $preview = 0
foreach($f in $targets){
  try {
    # 读入文本，统一 LF
    $raw = [System.IO.File]::ReadAllText($f.FullName)
    $lf = $raw -replace "`r`n","`n" -replace "`r","`n"
    if ($DryRun -and -not $Force){
      Write-Output "[DRYRUN] would convert to UTF-8+LF: $($f.FullName)"
      $preview++
      continue
    }
    if (-not $NoBackup){
      Copy-Item -LiteralPath $f.FullName -Destination ("$($f.FullName).utf8lf.bak") -ErrorAction SilentlyContinue | Out-Null
    }
    Write-Utf8NoBom -filePath $f.FullName -text $lf
    Write-Output "[CONVERTED] $($f.FullName)"
    $changed++
  } catch {
    Write-Warning "[ERROR] $($f.FullName): $($_.Exception.Message)"
    $errors++
  }
}

if ($DryRun -and -not $Force){
  Write-Output "[SUMMARY] preview=$preview errors=$errors"
  Write-Output "Hint: run with -Force to apply changes (creates .utf8lf.bak unless -NoBackup)"
} else {
  Write-Output "[SUMMARY] converted=$changed errors=$errors"
}
