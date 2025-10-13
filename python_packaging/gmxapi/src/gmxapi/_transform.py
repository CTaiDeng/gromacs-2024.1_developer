# SPDX-License-Identifier: GPL-3.0-only
#
# Copyright (C) 2022- The GROMACS Authors
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

"""A collection of data transformations to support shape/type normalization."""


def identity(x):
    return x


def broadcast(singular, width: int):
    """Fill an array of the given width from the singular data provided."""
    # Warning: the initial use case for this utility function is to repackage a Future of width 1 as a
    # Future of greater width, but this relies on an assumption about Future.result() that may change.
    # When *result()* is called, gmxapi *always* converts data of width 1 to a value of *dtype*, whereas a
    # Future of width > 1 gives its result as a list. See https://gitlab.com/gromacs/gromacs/-/issues/2993
    # and related issues.
    return [singular] * width
