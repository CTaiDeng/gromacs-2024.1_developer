# SPDX-License-Identifier: GPL-3.0-only
#
# Copyright (C) 2019- The GROMACS Authors
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

"""gmxapi Python package for GROMACS.

This package provides Python access to GROMACS molecular simulation tools.
Operations can be connected flexibly to allow high performance simulation and
analysis with complex control and data flows. Users can define new operations
in C++ or Python with the same tool kit used to implement this package.

"""

__all__ = [
    "commandline_operation",
    "concatenate_lists",
    "function_wrapper",
    "join_arrays",
    "logger",
    "logical_not",
    "make_constant",
    "mdrun",
    "modify_input",
    "ndarray",
    "read_tpr",
    "subgraph",
    "while_loop",
    "NDArray",
    "__version__",
]

from ._logging import logger
from .version import __version__

# Import utilities
from .operation import computed_result, function_wrapper

# Import public types
from .datamodel import NDArray

# Import the public operations
from .datamodel import ndarray
from .operation import concatenate_lists, join_arrays, logical_not, make_constant
from .commandline import commandline_operation
from .simulation import mdrun, modify_input, read_tpr

# TODO: decide where this lives
from .operation import subgraph

# TODO: decide where this lives
from .operation import while_loop
