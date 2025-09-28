@echo off
setlocal ENABLEDELAYEDEXPANSION

rem -------------------------------------------------
rem Setup Python virtualenv and install project deps
rem Default: envdir=.venv, install numpy + torch (CPU)
rem Flags:
rem   --envdir <dir>           Target virtualenv directory (default .venv)
rem   --requirements <path>    Install from requirements.txt first (optional)
rem   --no-torch               Do not install PyTorch (CPU)
rem   --tests                  Also install pytest (optional)
rem   --trace                  Print commands before running
rem -------------------------------------------------

set "ENVDIR=.venv"
set "REQUIREMENTS="
set "INSTALL_TORCH=1"
set "INSTALL_TESTS=0"
set "TRACE=0"

:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--envdir" (
  set "ENVDIR=%~2"
  shift & shift & goto parse_args
)
if /I "%~1"=="--requirements" (
  set "REQUIREMENTS=%~2"
  shift & shift & goto parse_args
)
if /I "%~1"=="--no-torch" (
  set "INSTALL_TORCH=0"
  shift & goto parse_args
)
if /I "%~1"=="--tests" (
  set "INSTALL_TESTS=1"
  shift & goto parse_args
)
if /I "%~1"=="--trace" (
  set "TRACE=1"
  shift & goto parse_args
)
echo Unknown argument: %~1
exit /b 1

:args_done

if "%TRACE%"=="1" echo [debug] trace enabled

rem Resolve Python (prefer py -3.10+, then py -3, then python)
set "PY_CMD="
for %%V in (3.12 3.11 3.10) do (
  if not defined PY_CMD (
    py -%%V -c "import sys" >NUL 2>&1 && set "PY_CMD=py -%%V"
  )
)
if not defined PY_CMD py -3 -c "import sys" >NUL 2>&1 && set "PY_CMD=py -3"
if not defined PY_CMD python -c "import sys" >NUL 2>&1 && set "PY_CMD=python"
if not defined PY_CMD (
  echo [error] Python 3.10+ not found. Please install Python or the Windows Python Launcher.
  exit /b 1
)

if "%TRACE%"=="1" echo [exec] %PY_CMD% -m venv "%ENVDIR%"
%PY_CMD% -m venv "%ENVDIR%"
if errorlevel 1 goto error

set "VENVPY=%ENVDIR%\Scripts\python.exe"
if not exist "%VENVPY%" (
  echo [error] virtualenv python not found: "%VENVPY%"
  goto error
)

call :run "%VENVPY%" -m pip install --upgrade pip setuptools wheel || goto error

if defined REQUIREMENTS (
  if "%TRACE%"=="1" echo [info] installing from requirements: "%REQUIREMENTS%"
  call :run "%VENVPY%" -m pip install --upgrade -r "%REQUIREMENTS%" || goto error
)

rem Project-required: numpy
call :run "%VENVPY%" -m pip install --upgrade numpy || goto error
"%VENVPY%" -c "import numpy as _; print('numpy ok')" >NUL 2>&1
if errorlevel 1 (
  echo [warn] numpy import failed, retrying with binary wheels...
  call :run "%VENVPY%" -m pip install --upgrade --only-binary=:all: numpy || goto error
  call :run "%VENVPY%" -c "import numpy as _; print('numpy ok')" || goto error
)

rem Optional: google-generativeai (用于提交信息生成)
call :run "%VENVPY%" -m pip install --upgrade google-generativeai || echo [warn] google-generativeai install failed
"%VENVPY%" -c "import importlib.metadata as im; print('google-generativeai:', im.version('google-generativeai'))" >NUL 2>&1 || echo [warn] google-generativeai import failed

rem Project-required: PyTorch (CPU), unless disabled
if "%INSTALL_TORCH%"=="1" (
  call :run "%VENVPY%" -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cpu
  set "TORCH_STATUS=%ERRORLEVEL%"
) else (
  set "TORCH_STATUS=999"
)

rem Optional: tests
if "%INSTALL_TESTS%"=="1" (
  call :run "%VENVPY%" -m pip install --upgrade pytest pytest-cov || goto error
)

echo [ok] versions:
call :run "%VENVPY%" -c "import sys,platform; print('Python:', sys.version); import numpy as np; print('numpy:', np.__version__)" || goto error

if "%TORCH_STATUS%"=="0" (
  call :run "%VENVPY%" -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())" || echo [warn] torch import failed unexpectedly
) else (
  if "%INSTALL_TORCH%"=="1" echo [warn] torch installation may be unsupported on this Python; skipped reporting
)

echo.
echo [done] Activate your environment:
echo    %ENVDIR%\Scripts\activate
exit /b 0

:run
if "%TRACE%"=="1" echo [exec] %*
%*
exit /b %ERRORLEVEL%

:error
echo [fail] setup failed (errorlevel=%ERRORLEVEL%)
exit /b 1

