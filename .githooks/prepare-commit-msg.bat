@echo off
setlocal ENABLEDELAYEDEXPANSION
rem Cross-platform Git hook wrapper for Windows (prepare-commit-msg)
rem Usage: prepare-commit-msg <msgfile> [commit_source] [sha]

set MSG_FILE=%~1
set COMMIT_SOURCE=%~2

rem Skip for merge/squash
if /I "%COMMIT_SOURCE%"=="merge" exit /b 0
if /I "%COMMIT_SOURCE%"=="squash" exit /b 0

rem Read current message (strip comment lines starting with #)
set CONTENT=
for /f "usebackq tokens=* delims=" %%L in (`type "%MSG_FILE%" ^| findstr /r /v "^#"`) do (
  if defined CONTENT (
    set CONTENT=!CONTENT!%%L
  ) else (
    set CONTENT=%%L
  )
)
set CONTENT=%CONTENT: =%

set NEED_GEN=0
if "%CONTENT%"=="" set NEED_GEN=1
if /I "%CONTENT%"=="update" set NEED_GEN=1

if %NEED_GEN%==1 (
  for /f "usebackq delims=" %%R in (`git rev-parse --show-toplevel 2^>NUL`) do set REPO_ROOT=%%R
  if not defined REPO_ROOT set REPO_ROOT=%CD%

  rem Prefer py launcher, then python3, then python
  set PY=
  where py >NUL 2>&1 && set PY=py -3
  if not defined PY ( where python3 >NUL 2>&1 && set PY=python3 )
  if not defined PY ( where python >NUL 2>&1 && set PY=python )

  if defined PY (
    for /f "usebackq delims=" %%O in (`"%PY%" "%REPO_ROOT%\my_scripts\gen_commit_msg_googleai.py" 2^>NUL`) do (
      >"%MSG_FILE%" echo %%O
      goto :done
    )
    rem Fallback placeholder
    >"%MSG_FILE%" echo update
  ) else (
    >"%MSG_FILE%" echo update
  )
)

:done
endlocal & exit /b 0

