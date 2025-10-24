@echo off
setlocal

REM Wrapper for PowerShell converter: convert_to_utf8_lf.ps1
REM Usage:
REM   convert_to_utf8_lf.cmd [path]
REM If [path] is omitted, current directory is used.

set "SCRIPT_DIR=%~dp0"
set "TARGET=%~1"
if not defined TARGET set "TARGET=."

pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%convert_to_utf8_lf.ps1" "%TARGET%" -Force

endlocal
exit /b %ERRORLEVEL%

