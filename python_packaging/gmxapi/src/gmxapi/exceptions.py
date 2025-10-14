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

"""
Exceptions and Warnings raised by gmxapi module operations.

Errors, warnings, and other exceptions used in the GROMACS
Python package are defined in the `exceptions` submodule.

The gmxapi Python package defines a root exception,
exceptions.Error, from which all Exceptions thrown from
within the module should derive. If a published component of
the gmxapi package throws an exception that cannot be caught
as a gmxapi.exceptions.Error, please report the bug.
"""

__all__ = [
    "ApiError",
    "DataShapeError",
    "Error",
    "FeatureNotAvailableError",
    "MissingImplementationError",
    "ProtocolError",
    "TypeError",
    "UsageError",
    "ValueError",
    "Warning",
]


class Error(Exception):
    """Base exception for gmx.exceptions classes."""


class Warning(Warning):
    """Base warning class for gmx.exceptions."""


class ApiError(Error):
    """An API operation was attempted with an incompatible object."""


class DataShapeError(Error):
    """An object has an incompatible shape.

    This exception does not imply that the Type or any other aspect of the data
    has been checked.
    """


class MissingImplementationError(Error):
    """Specified feature is not implemented in the current code.

    This exception indicates that the implemented code does not support the
    API as specified. The calling code has used valid syntax, as documented for
    the API, but has reached incompletely implemented code, which should be
    considered a bug.

    .. versionchanged:: 0.3
        Named changed to avoid conflict with built-in :py:class:`NotImplementedError` exception
    """

    # May be useful for error checking in base classes or as a development tool
    # to avoid releasing incomplete implementations (e.g. overlooked "To do"s)


class FeatureNotAvailableError(Error):
    """Requested feature not available in the current environment.

    This exception will usually indicate an issue with the user's environment or
    run time details. There may be a missing optional dependency, which should
    be specified in the exception message.
    """


class ProtocolError(Error):
    """Unexpected API behavior or protocol violation.

    This exception generally indicates a gmxapi bug, since it should only
    occur through incorrect assumptions or misuse of API implementation internals.
    """


class TypeError(Error):
    """Incompatible type for gmxapi data.

    Reference datamodel.rst for more on gmxapi data typing.
    """


class UsageError(Error):
    """Unsupported syntax or call signatures.

    Generic usage error for gmxapi module.
    """


class ValueError(Error):
    """A user-provided value cannot be interpreted or doesn't make sense."""
