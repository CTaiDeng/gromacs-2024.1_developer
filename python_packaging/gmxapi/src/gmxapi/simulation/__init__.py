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

"""GROMACS simulation subpackage for gmxapi.

Provides operations for configuring and running molecular simulations.

The initial version of this module is a port of the gmxapi 0.0.7 facilities from
https://github.com/kassonlab/gmxapi and is not completely integrated with the
gmxapi 0.1 specification. Operation execution is dispatched to the old execution
manager for effective ensemble handling and C++ MD module binding. This should
be an implementation detail that is not apparent to the typical user, but it is
worth noting that chains of gmxapi.simulation module operations will be
automatically bundled for execution as gmxapi 0.0.7 style API sessions. Run time
options and file handling will necessarily change as gmxapi data flow handling
evolves.

In other words, if you rely on behavior not specified explicitly in the user
documentation, please keep an eye on the module documentation when updating
gmxapi and please participate in the ongoing discussions for design and
implementation.
"""

__all__ = ["abc", "mdrun", "modify_input", "read_tpr"]

from gmxapi.simulation import abc
from .mdrun import mdrun
from .read_tpr import read_tpr
from .modify_input import modify_input
