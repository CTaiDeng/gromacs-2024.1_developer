<#
  Copyright (C) 2025 GaoZheng
  SPDX-License-Identifier: GPL-3.0-only
  本脚本用于本仓库：仅更新 README.md 中的“## project_docs”索引。
  行为：扫描 `my_docs/project_docs`（非递归，排除 `kernel_reference`），
  为每个 Markdown 文件生成如下两段式条目（注意中间有空行）：

  - `my_docs/project_docs/<file>.md`：

    <从“## 摘要”节提取的前 N 个字；若无摘要则留白>

  仅截取“## 摘要”章节的内容，超出该章节的不取；字符数由参数控制。
#>
param(
  [int]$MaxChars = 300,
  [string]$ReadmePath = 'README.md',
  [string]$DocsDir = 'my_docs/project_docs'
)

[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$ErrorActionPreference = 'Stop'

function Get-AbstractFromSection([string]$path, [int]$maxChars){
  try { $raw = Get-Content -LiteralPath $path -Raw -Encoding UTF8 } catch { return '' }
  # 仅提取以“## 摘要”开始至下一个标题或文件尾的内容
  $m = [Regex]::Match($raw, '(?ms)^\s*##\s*摘要\s*$\s*([\s\S]*?)(?=^\s*#{1,6}\s|\z)')
  if(-not $m.Success){ return '' }
  $text = $m.Groups[1].Value
  # 清理常见 Markdown 结构
  $text = $text -replace '(?ms)```.*?```',''              # 代码块
  $text = $text -replace '(?m)^\s*`{3,}.*$',''           # 孤立栅栏
  $text = $text -replace '!\[[^\]]*\]\([^)]*\)',''    # 图片
  $text = [Regex]::Replace($text, '\[([^\]]+)\]\([^)]*\)', '$1') # 链接文本
  $text = $text -replace '`',''                           # 行内代码
  $text = $text -replace '(?m)^\s*#{1,6}\s*',''         # 标题标记
  $text = $text -replace '(?m)^\s*>\s*',''              # 引用前缀
  $text = $text -replace '(?m)^\s*[-*_]{3,}\s*$',''     # 分隔线
  $text = ($text -split "\r?\n" | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne '' }) -join ' '
  $text = ($text -replace '\s+',' ').Trim()
  if([string]::IsNullOrWhiteSpace($text)){ return '' }
  if($text.Length -gt $maxChars){ $text = $text.Substring(0, $maxChars).Trim() }
  return $text
}

function BuildProjectDocsIndex([string]$docsDir, [int]$maxChars){
  $root = (Resolve-Path '.').Path
  $dir  = Join-Path $root $docsDir
  if(-not (Test-Path -LiteralPath $dir -PathType Container)){
    throw "Docs directory not found: $docsDir"
  }
  $files = Get-ChildItem -LiteralPath $dir -File -Filter '*.md' | Where-Object {
    $_.Name -ne 'LICENSE.md' -and $_.FullName -notmatch '\\kernel_reference\\'
  } | Sort-Object Name

  $lines = New-Object System.Collections.Generic.List[string]
  foreach($f in $files){
    $rel = ($f.FullName | Resolve-Path -Relative)
    $rel = $rel -replace '^\.+\\',''         # 去掉 .\ 前缀
    $rel = $rel -replace '/', '\\'
    $lines.Add('- `' + $rel + '`：')
    $abs = Get-AbstractFromSection -path $f.FullName -maxChars $maxChars
    if(-not [string]::IsNullOrWhiteSpace($abs)){
      $lines.Add('  ' + $abs)
    } else {
      # 没有摘要章节：留白（不填充摘要文本）
    }
  }
  return $lines
}

function UpdateReadmeProjectDocs([string]$readmePath, [System.Collections.Generic.List[string]]$indexLines){
  if(-not (Test-Path -LiteralPath $readmePath -PathType Leaf)){
    throw "README not found: $readmePath"
  }
  $all = Get-Content -LiteralPath $readmePath -Encoding UTF8
  $start = -1; $end = $all.Length
  for($i=0; $i -lt $all.Length; $i++){
    if($all[$i] -match '^\s*##\s*project_docs\s*$'){ $start = $i; break }
  }
  if($start -lt 0){
    # 若无该小节，则在末尾追加
    $out = New-Object System.Collections.Generic.List[string]
    foreach($ln in $all){ $out.Add($ln) }
    if($out.Count -gt 0 -and $out[$out.Count-1] -ne ''){ $out.Add('') }
    $out.Add('## project_docs')
    $out.AddRange($indexLines)
    $out | Set-Content -LiteralPath $readmePath -Encoding UTF8
    return
  }
  for($j=$start+1; $j -lt $all.Length; $j++){
    if($all[$j] -match '^\s*##\s+' ){ $end = $j; break }
  }
  $pre  = if($start -gt 0) { $all[0..$start] } else { @($all[$start]) }
  $post = if($end -lt $all.Length) { $all[$end..($all.Length-1)] } else { @() }
  $out  = New-Object System.Collections.Generic.List[string]
  foreach($ln in $pre){ $out.Add($ln) }
  # 用新索引替换该小节下的全部内容
  $out.AddRange($indexLines)
  foreach($ln in $post){ $out.Add($ln) }
  $out | Set-Content -LiteralPath $readmePath -Encoding UTF8
}

$index = BuildProjectDocsIndex -docsDir $DocsDir -maxChars $MaxChars
UpdateReadmeProjectDocs -readmePath $ReadmePath -indexLines $index
Write-Host "Updated $ReadmePath project_docs index with $($index.Count) lines."

