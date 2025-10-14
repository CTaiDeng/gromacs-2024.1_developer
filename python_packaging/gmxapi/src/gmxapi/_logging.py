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

"""Python logging facilities use the built-in logging module.

Upon import, the gmxapi package sets a placeholder "NullHandler" to block
propagation of log messages to the root logger (and sys.stderr, if not handled).

If you want to see gmxapi logging output on `sys.stderr`, import `logging` in your
script or module and configure it. For the simplest case, consider
`logging.basicConfig`::

    >>> import logging
    >>> logging.basicConfig(level=logging.DEBUG)

For more advanced usage, consider attaching a
`logging.StreamHandler` to the ``gmxapi`` logger.

The gmxapi logging module adds an additional ``rank_tag`` log formatter field that can
be particularly helpful in ensemble MPI workflows.

Example::

    ch = logging.StreamHandler()
    # Optional: Set log level.
    ch.setLevel(logging.DEBUG)
    # Optional: create formatter and add to character stream handler
    formatter = logging.Formatter('%(levelname)s %(asctime)s:%(name)s %(rank_tag)s%(message)s')
    ch.setFormatter(formatter)
    # add handler to logger
    logging.getLogger('gmxapi').addHandler(ch)

To handle log messages that are issued while importing :py:mod:`gmxapi` and its submodules,
attach the handler before importing :py:mod:`gmxapi`

Each module in the gmxapi package uses its own hierarchical logger to allow
granular control of log handling (e.g. ``logging.getLogger('gmxapi.operation')``).
Refer to the Python :py:mod:`logging` module for information on connecting to and handling
logger output.
"""

__all__ = ["logger"]

# Import system facilities
import logging
from logging import getLogger, DEBUG, NullHandler

# Define `logger` attribute that is used by submodules to create sub-loggers.
logger = getLogger("gmxapi")
# Prevent gmxapi logs from reaching logging.lastResort (and printing to sys.stderr)
# if the user does not take action to handle logging.
logger.addHandler(NullHandler(level=DEBUG))

logger.info("Importing gmxapi.")
