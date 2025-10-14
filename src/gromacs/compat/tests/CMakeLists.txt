# SPDX-License-Identifier: GPL-3.0-only
#
# Copyright (C) 2012- The GROMACS Authors
# Copyright (C) 2025 GaoZheng
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# ---
#
# This file is part of a modified version of the GROMACS molecular simulation package.
# For details on the original project, consult https://www.gromacs.org.
#
# To help fund GROMACS development, we humbly ask that you cite
# the research papers on the package. Check out https://www.gromacs.org.

if (GMX_BUILD_UNITTESTS)
    gmx_add_unit_test(CompatibilityHelpersTests compat-test
        CPP_SOURCE_FILES
            mp11.cpp
            pointers.cpp
            )
    # Maintainer note: The files here may be borrowed from other projects, and
    # can require additional handling for compiler flags, warning suppression,
    # etcetera. Rather than documenting best practices and tips here, we refer
    # the reader to the `git` repository history for examples of how integration
    # caveats have been handled in the past (for borrowed code that has since
    # been removed when it became unnecessary).
endif()
