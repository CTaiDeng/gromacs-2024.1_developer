# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

<#
安全的 UTF-8+LF 批量转换脚本。
- 统一将文本/源码文件重写为 UTF-8（无 BOM）+ LF 行尾。
- 默认 DryRun 预览，需 -Force 才实际落盘；保留 .utf8lf.bak 备份（可 -NoBackup 关闭）。
- 跳过常见二进制与大目录（.git、node_modules、out、.venv、kernel_reference）。
- 支持 -Verbose 输出调试信息，及 -TraceLines 逐行打印（带行号与 EOL 类型）。

示例：
  # 预览将被转换的文件
  pwsh -File my_scripts/convert_to_utf8_lf.ps1 -DryRun -Verbose

  # 实际执行（保留备份）
  pwsh -File my_scripts/convert_to_utf8_lf.ps1 -Force

  # 指定目录/扩展名，不保留备份，并逐行打印
  pwsh -File my_scripts/convert_to_utf8_lf.ps1 -Path . -Extensions .py,.md,.json -Force -NoBackup -TraceLines -Verbose
#>

[CmdletBinding()]
param(
  [string]$Path = '.',
  [string[]]$Extensions = @(
    '.md','.txt','.json','.yml','.yaml','.toml','.ini','.cfg','.ps1','.psm1','.psd1',
    '.py','.sh','.bat','.cmd','.c','.h','.hpp','.cc','.cpp','.cu','.cl','.cmake','.ts','.tsx','.js','.jsx','.css','.scss','.html'
  ),
  [switch]$DryRun,
  [switch]$Force,
  [switch]$NoBackup,
  [switch]$TraceLines,
  [int]$MaxLineLength = 200
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

Write-Verbose ("[DEBUG] scan_path={0} candidates={1}" -f $root, $targets.Count)

$changed = 0; $errors = 0; $preview = 0
foreach($f in $targets){
  try {
    # 读入文本
    $raw = [System.IO.File]::ReadAllText($f.FullName)
    # Debug: 基本统计
    $hasCRLF = $raw -match "`r`n"
    $hasSoloCR = $raw -match "`r(?!`n)"
    $hasLF = $raw -match "`n"
    $lineCount = ([regex]::Matches($raw, "\r\n|\n|\r").Count) + 1
    Write-Verbose ("[DEBUG] file={0} hasCRLF={1} hasCRonly={2} hasLF={3} approx_lines={4}" -f $f.FullName,$hasCRLF,$hasSoloCR,$hasLF,$lineCount)

    if ($TraceLines) {
      Write-Verbose ("[DEBUG] --- TRACE BEGIN: {0}" -f $f.FullName)
      $rx = [regex]"(.*?)(\r\n|\n|\r|$)"
      $m = $rx.Matches($raw)
      $idx = 1
      foreach($x in $m){
        $txt = $x.Groups[1].Value
        $eol = $x.Groups[2].Value
        if ($null -ne $txt -and $txt.Length -gt $MaxLineLength){
          $txt = $txt.Substring(0,$MaxLineLength) + '...'
        }
        $eolName = switch ($eol) { "`r`n" { 'CRLF' } "`n" { 'LF' } "`r" { 'CR' } default { 'EOF' } }
        Write-Host ("[TRACE] line {0} eol={1}: {2}" -f $idx,$eolName,$txt)
        $idx++
      }
      Write-Verbose ("[DEBUG] --- TRACE END: {0}" -f $f.FullName)
    }

    # 统一 LF
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

