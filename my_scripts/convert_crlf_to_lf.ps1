<#
兼容包装脚本（已弃用）：调用 convert_to_utf8_lf.ps1，保持旧名可用。
用法与参数保持一致；建议改用新脚本名。
#>

param([Parameter(ValueFromRemainingArguments=$true)]$ArgsPassThru)

$scriptPath = Join-Path -Path $PSScriptRoot -ChildPath 'convert_to_utf8_lf.ps1'
if (-not (Test-Path -LiteralPath $scriptPath)){
  Write-Error "convert_to_utf8_lf.ps1 not found at $scriptPath"; exit 1
}
Write-Host "[DEPRECATED] Use my_scripts/convert_to_utf8_lf.ps1 instead. Forwarding..." -ForegroundColor Yellow
& $scriptPath @ArgsPassThru

