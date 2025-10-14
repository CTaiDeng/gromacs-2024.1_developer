# SPDX-License-Identifier: GPL-3.0-only
#
# Copyright (C) 2014- The GROMACS Authors
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

set(_gmx_find_args "")
if (GROMACS_FIND_VERSION)
    if (GROMACS_FIND_VERSION VERSION_LESS "5.1")
        message(FATAL_ERROR
            "This version of FindGROMACS.cmake requires GROMACS-provided "
            "package configuration files, and only works to find "
            "GROMACS 5.1 or later.")
    endif()
    list(APPEND _gmx_find_args ${GROMACS_FIND_VERSION})
    if (GROMACS_FIND_VERSION_EXACT)
        list(APPEND _gmx_find_args EXACT)
    endif()
endif()
if (GROMACS_FIND_REQUIRED)
    list(APPEND _gmx_find_args REQUIRED)
endif()
if (GROMACS_FIND_QUIETLY)
    list(APPEND _gmx_find_args QUIET)
endif()

# Determine the actual name of the package configuration files.
set(_gmx_pkg_name gromacs)
if (DEFINED GROMACS_SUFFIX)
    set(_gmx_pkg_name gromacs${GROMACS_SUFFIX})
endif()
# Delegate all the actual work to the package configuration files.
# The CONFIGS option is not really necessary, but provides a bit better error
# messages, since we actually know what the config file should be called.
find_package(GROMACS ${_gmx_find_args} CONFIG
             NAMES ${_gmx_pkg_name}
             CONFIGS ${_gmx_pkg_name}-config.cmake)
unset(_gmx_find_args)
unset(_gmx_pkg_name)
