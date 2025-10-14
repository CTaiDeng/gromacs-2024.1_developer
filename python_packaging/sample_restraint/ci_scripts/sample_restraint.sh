#!/usr/bin/env bash
# Copyright (C) 2025 GaoZheng
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of this project.
# Licensed under the GNU General Public License version 3.
# See https://www.gnu.org/licenses/gpl-3.0.html for details.
set -ev

rm -rf build
mkdir build
pushd build
 cmake .. -DPYTHON_EXECUTABLE=$PYTHON
 make -j2 install
 make -j2 test
 $PYTHON -c "import myplugin"
popd
pushd tests
 $PYTHON -m pytest
 mpiexec -n 2 $PYTHON -m mpi4py -m pytest --log-cli-level=DEBUG -s --verbose
popd
