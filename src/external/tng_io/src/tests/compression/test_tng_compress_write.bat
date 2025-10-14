REM Copyright (C) 2025 GaoZheng
REM SPDX-License-Identifier: GPL-3.0-only
REM This file is part of this project.
REM Licensed under the GNU General Public License version 3.
REM See https://www.gnu.org/licenses/gpl-3.0.html for details.
@echo off
setlocal enableextensions enabledelayedexpansion
SET /A I=0
:start
SET /A I+=1
test_tng_compress_gen%I%
IF "%I%" == "78" (
  GOTO end
) ELSE (
  GOTO start
)
:end
endlocal
