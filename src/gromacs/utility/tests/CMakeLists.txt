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

gmx_add_unit_test(UtilityUnitTests utility-test
    CPP_SOURCE_FILES
        alignedallocator.cpp
        arrayref.cpp
        booltype.cpp
        bitmask32.cpp bitmask64.cpp bitmask128.cpp
        cstringutil.cpp
        defaultinitializationallocator.cpp
        enumerationhelpers.cpp
        fixedcapacityvector.cpp
        inmemoryserializer.cpp
        keyvaluetreeserializer.cpp
        keyvaluetreetransform.cpp
        listoflists.cpp
        logger.cpp
        message_string_collector.cpp
        path.cpp
        physicalnodecommunicator.cpp
        range.cpp
        strconvert.cpp
        stringtoenumvalueconverter.cpp
        stringutil.cpp
        template_mp.cpp
        textreader.cpp
        textwriter.cpp
        typetraits.cpp
        )
# TODO: Remove `legacy_modules` once specific modules are explicitly linked.
target_link_libraries(utility-test PRIVATE legacy_modules)
target_link_libraries(utility-test PRIVATE utility)

gmx_add_mpi_unit_test(UtilityMpiUnitTests utility-mpi-test 2
    CPP_SOURCE_FILES
        physicalnodecommunicator_mpi.cpp
        )
