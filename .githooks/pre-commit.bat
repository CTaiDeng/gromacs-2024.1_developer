REM Copyright (C) 2025 GaoZheng
REM SPDX-License-Identifier: GPL-3.0-only
REM This file is part of this project.
REM Licensed under the GNU General Public License version 3.
REM See https://www.gnu.org/licenses/gpl-3.0.html for details.
@echo off
REM UTF-8 (no BOM) + LF checker wrapper for Windows
REM Delegates to the POSIX shell hook when available; read-only validation.

where sh >nul 2>nul
if %errorlevel%==0 (
  sh .githooks/pre-commit
  exit /b %errorlevel%
)

REM Fallback: do nothing (no sh available)
exit /b 0
