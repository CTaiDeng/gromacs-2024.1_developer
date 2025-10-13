@echo off
setlocal ENABLEDELAYEDEXPANSION
rem Cross-platform Git hook wrapper for Windows (pre-commit)

rem Resolve repo root
for /f "usebackq delims=" %%R in (`git rev-parse --show-toplevel 2^>NUL`) do set REPO_ROOT=%%R
if not defined REPO_ROOT set REPO_ROOT=%CD%

rem Pick Python
set PY=python
where python3 >NUL 2>&1 && set PY=python3
where py >NUL 2>&1 && set PY=py -3

rem 1) Align documents (non-blocking)
"%PY%" "%REPO_ROOT%\my_scripts\align_my_documents.py"
if errorlevel 1 echo [pre-commit] align_my_documents.py returned non-zero, continuing...

rem Stage any changes to docs (renames/content fixes)
git add -A

rem 2) Compliance guard (blocking)
"%PY%" "%REPO_ROOT%\my_scripts\check_derivation_guard.py"
set EXITCODE=%ERRORLEVEL%

endlocal & exit /b %EXITCODE%

