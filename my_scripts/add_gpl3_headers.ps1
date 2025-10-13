#!/usr/bin/env pwsh
# Add GPL-3.0 header to source files in this repo
# Usage examples:
#   pwsh my_scripts/add_gpl3_headers.ps1
#   pwsh my_scripts/add_gpl3_headers.ps1 -Paths src,api,include -WhatIf

[CmdletBinding(SupportsShouldProcess=$true)]
param(
  [string[]]$Paths = @('src','api','include','admin','scripts','tests','python_packaging'),
  [string[]]$Extensions = @(
    '.c','.cc','.cpp','.cxx','.h','.hh','.hpp','.hxx','.cu','.cuh',
    '.py','.sh','.ps1','.psm1','.cmake','.bat','.cmd','.js','.ts',
    '.java','.rs','.go','.m','.mm','.R'
  ),
  [switch]$Recurse = $true,
  [ValidateSet('GPL-3.0-only','GPL-3.0-or-later')]
  [string]$Spdx = 'GPL-3.0-only'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-RepoRoot {
  try {
    $gitTop = (git rev-parse --show-toplevel) 2>$null
    if ($LASTEXITCODE -eq 0 -and $gitTop) { return (Resolve-Path $gitTop).Path }
  } catch {}
  return (Resolve-Path "$PSScriptRoot/.." ).Path
}

$RepoRoot = Get-RepoRoot
Push-Location $RepoRoot
try {
  # Exclusions (paths relative to repo root, case-insensitive)
  $ExcludePaths = @(
    '.git', '.venv', '.pip-cache', 'cmake-build-release-visual-studio-2022', 'out', 'out/', 'logs',
    'my_docs', 'my_docs/', 'my_project/*/LIG.acpype', 'my_docs/project_docs/kernel_reference',
    'res', 'share', 'out', 'out.txt'
  )

  # Files to never touch (explicit whitelist-exclude)
  $NeverTouch = @(
    'my_docs/project_docs/LICENSE.md',
    'my_project/gmx_split_20250924_011827/docs/LICENSE.md'
  )

  function Should-ExcludePath($rel) {
    $r = $rel.ToLowerInvariant()
    foreach ($p in $ExcludePaths) {
      $pp = $p.ToLowerInvariant()
      if ($pp.EndsWith('/*')) {
        $prefix = $pp.Substring(0, $pp.Length-2)
        if ($r.StartsWith($prefix.TrimEnd('/'))) { return $true }
      }
      if ($r -eq $pp.TrimEnd('/')) { return $true }
      if ($r.StartsWith($pp.TrimEnd('/') + '/')) { return $true }
    }
    foreach ($nt in $NeverTouch) {
      $nt = $nt.ToLowerInvariant()
      if ($r -eq $nt) { return $true }
    }
    return $false
  }

  function Detect-Style($file) {
    $name = [IO.Path]::GetFileName($file)
    $ext = [IO.Path]::GetExtension($file).ToLowerInvariant()
    if ($name -ieq 'CMakeLists.txt') { return '#'}
    switch ($ext) {
      { @('.c','.cc','.cpp','.cxx','.h','.hh','.hpp','.hxx','.cu','.cuh','.java','.js','.ts','.m','.mm') -contains $_ } { return 'block' }
      '.bat' { return 'bat' }
      '.cmd' { return 'bat' }
      default { return '#'}
    }
  }

  function Make-HeaderLines($style) {
    $year = '2025'
    $copyLine = "Copyright (C) $year GaoZheng"
    $spdxLine = "SPDX-License-Identifier: $Spdx"
    $gpl = @(
      $copyLine,
      $spdxLine,
      'This file is part of this project.',
      'Licensed under the GNU General Public License version 3.',
      'See https://www.gnu.org/licenses/gpl-3.0.html for details.'
    )
    switch ($style) {
      'block' {
        $lines = @('/*')
        foreach ($l in $gpl) { $lines += " * $l" }
        $lines += ' */'
        return $lines
      }
      'bat' {
        return ($gpl | ForEach-Object { 'REM ' + $_ })
      }
      default {
        return ($gpl | ForEach-Object { '# ' + $_ })
      }
    }
  }

  function Already-HasHeader($text) {
    $head = -join ($text | Select-Object -First 60)
    if ($head -match 'SPDX-License-Identifier:\s*GPL-3\.0') { return $true }
    if ($head -match 'GNU\s+General\s+Public\s+License' -and $head -match 'version\s*3') { return $true }
    if ($head -match 'GPL-3\.0') { return $true }
    return $false
  }

  function Insert-Header($path) {
    $style = Detect-Style $path
    $lines = Get-Content -LiteralPath $path -Raw -ErrorAction Stop
    $crlf = ($lines -match "`r`n")
    $eol = if ($crlf) { "`r`n" } else { "`n" }
    $arr = $lines -split "`r?`n"
    if (Already-HasHeader $arr) { return $false }

    $header = Make-HeaderLines $style
    $insertAt = 0
    # Preserve shebang and python encoding line
    if ($arr.Count -gt 0 -and $arr[0].StartsWith('#!')) { $insertAt = 1 }
    if ($arr.Count -gt ($insertAt) -and $arr[$insertAt] -match 'coding\s*[:=]\s*utf-?8') { $insertAt += 1 }

    $before = $arr[0..($insertAt-1)]
    $after = $arr[$insertAt..($arr.Count-1)]
    $new = @()
    if ($before) { $new += $before }
    $new += $header
    if ($after) { $new += $after }
    $content = ($new -join $eol)
    if (-not $content.EndsWith($eol)) { $content += $eol }
    Set-Content -LiteralPath $path -Value $content -Encoding utf8
    return $true
  }

  $extSet = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
  foreach ($e in $Extensions) { [void]$extSet.Add($e) }

  $scanned = 0; $updated = 0; $skipped = 0
  foreach ($p in $Paths) {
    $abs = Join-Path $RepoRoot $p
    if (-not (Test-Path $abs)) { continue }
    $files = if ($Recurse) { Get-ChildItem -LiteralPath $abs -Recurse -File -ErrorAction SilentlyContinue } else { Get-ChildItem -LiteralPath $abs -File -ErrorAction SilentlyContinue }
    foreach ($f in $files) {
      $rel = (Resolve-Path -LiteralPath $f.FullName).Path.Substring($RepoRoot.Length).TrimStart('\\','/') -replace '\\','/'
      if (Should-ExcludePath $rel) { $skipped++; continue }
      $ext = [IO.Path]::GetExtension($f.Name)
      if (-not $extSet.Contains($ext)) { $skipped++; continue }
      $scanned++
      if ($PSCmdlet.ShouldProcess($rel, 'Insert GPL-3.0 header')) {
        try {
          if (Insert-Header $f.FullName) { $updated++ }
        } catch {
          Write-Warning "Failed: $rel - $_"
          $skipped++
        }
      }
    }
  }

  Write-Host "[gpl-headers] scanned=$scanned updated=$updated skipped=$skipped"
} finally {
  Pop-Location
}

