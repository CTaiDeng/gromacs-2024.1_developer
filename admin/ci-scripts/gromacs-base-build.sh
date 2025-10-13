#!/usr/bin/env bash
# Copyright (C) 2025- GaoZheng
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of this project.
# Licensed under the GNU General Public License version 3.
# See https://www.gnu.org/licenses/gpl-3.0.html for details.
set -e
CMAKE=${CMAKE:-$(which cmake)}
cd $BUILD_DIR
$CMAKE --build . -- -j$KUBERNETES_CPU_LIMIT 2>&1 | tee buildLogFile.log
$CMAKE --build . --target tests -- -j$KUBERNETES_CPU_LIMIT 2>&1 | tee testBuildLogFile.log

# Find compiler warnings
awk '/warning/,/warning.*generated|^$/' buildLogFile.log testBuildLogFile.log \
      | grep -v "CMake" | tee buildErrors.log || true
grep "cannot be built" buildLogFile.log testBuildLogFile.log | tee -a buildErrors.log || true
grep "fatal error" buildLogFile.log testBuildLogFile.log | tee -a buildErrors.log || true
grep "error generated when compiling" buildLogFile.log testBuildLogFile.log | tee -a buildErrors.log || true
grep "error:" buildLogFile.log testBuildLogFile.log | tee -a buildErrors.log || true

# Find linking errors:
grep "^/usr/bin/ld:" buildLogFile.log testBuildLogFile.log | tee -a buildErrors.log || true

# Install GROMACS
set +e -o pipefail # Make "$?" work correctly
$CMAKE --build . --target install 2>&1 | tee installBuildLogFile.log
EXITCODE=$?
set -e +o pipefail

# Fail if there were warnings or errors reported
if [ -s buildErrors.log ] || [ $EXITCODE != 0 ] ; then echo "Found compiler warning during build"; cat buildErrors.log; exit 1; fi
# Remove object files to minimize artifact size
find . -mindepth 1 -name '*.o' ! -type l -printf '%p\n' -delete 2>&1 > remove-build-objects.log
cd ..
