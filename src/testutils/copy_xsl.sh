#!/bin/sh
# Copyright (C) 2025 GaoZheng
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of this project.
# Licensed under the GNU General Public License version 3.
# See https://www.gnu.org/licenses/gpl-3.0.html for details.

# Change to root of the source tree, no matter from where the script is
# invoked.
cd `dirname $0`
cd ../../

for destdir in analysisdata selection trajectoryanalysis ; do
    cp -f src/testutils/common-referencedata.xsl \
          src/gromacs/$destdir/tests/refdata/
done

for destdir in trajectoryanalysis ; do
    cp -f src/gromacs/analysisdata/tests/refdata/analysisdata-referencedata.xsl \
          src/gromacs/$destdir/tests/refdata/
done
