# SPDX-License-Identifier: GPL-3.0-only
#
# Copyright (C) 2019- The GROMACS Authors
# Copyright (C) 2025- GaoZheng
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

"""Test exception propagation and inheritance (catchability)"""

# We do not import any gmxapi modules at file scope because one thing we are
# checking for is hidden dependencies.
import pytest


@pytest.mark.usefixtures("cleandir")
def test_catchability():
    """Test exception inheritance of C++ extension module.

    All exceptions should be catchable with the package base exception.
    """
    from gmxapi._gmxapi import Exception as ExtensionException
    import gmxapi

    with pytest.raises(gmxapi.exceptions.Error):
        raise ExtensionException()
    from gmxapi._gmxapi import UnknownException as ExtensionException

    with pytest.raises(gmxapi.exceptions.Error):
        raise ExtensionException()
