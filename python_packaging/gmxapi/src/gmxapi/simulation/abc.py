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

"""
Abstract base classes for gmxapi.simulation module
==================================================

These classes allow static and dynamic type checking for gmxapi Python
interfaces. Some may be used as bases to inherit behaviors, but users should
not assume that a gmxapi object actually inherits from classes in this module.

For more information on the concept of abstract base classes in Python, refer
to https://docs.python.org/3/library/abc.html

For more on type hinting, see https://docs.python.org/3/library/typing.html
"""

import abc


class ModuleObject(abc.ABC):
    """Extended interface for objects in the gmaxpi.simulation module.

    Implies availability of additional binding details.
    """
