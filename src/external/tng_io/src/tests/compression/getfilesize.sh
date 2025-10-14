#!/bin/sh
# Copyright (C) 2025 GaoZheng
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of this project.
# Licensed under the GNU General Public License version 3.
# See https://www.gnu.org/licenses/gpl-3.0.html for details.
if [ -z "$3" ]; then
    echo $0 STARTTEST ENDTEST TNGFILEDIR
else
    STARTTEST="$1"
    ENDTEST="$2"
    TNGFILEDIR="$3"
    for testnum in $(seq $STARTTEST $ENDTEST); do
	if [ -r $TNGFILEDIR/test$testnum.tng_compress ]; then
	    grep -v "EXPECTED_FILESIZE" test$testnum.h >tmp$$.h
	    echo "#define EXPECTED_FILESIZE" $(ls -l $TNGFILEDIR/test$testnum.tng_compress |awk '{print $5}'). >>tmp$$.h
	    mv tmp$$.h test$testnum.h
	fi
    done
fi
