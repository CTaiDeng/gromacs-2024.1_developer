#!/bin/sh
# Copyright (C) 2025- GaoZheng
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of this project.
# Licensed under the GNU General Public License version 3.
# See https://www.gnu.org/licenses/gpl-3.0.html for details.
numtests=78
for x in $(seq 1 $numtests); do
    ./test_tng_compress_read$x
done
