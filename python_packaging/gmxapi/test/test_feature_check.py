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

"""Test feature checking API and optional feature handling."""

from packaging.version import parse

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

import pytest
import gmxapi
import gmxapi._gmxapi as core
from gmxapi.utility import config


@pytest.mark.usefixtures("cleandir")
def test_feature_check():
    """Check the query features of the binary extension module."""
    assert hasattr(core, "has_feature")
    try:
        gmxpy_version = parse(version("gmxapi"))
    except PackageNotFoundError:
        # If the package distribution has not been built (such as when it has
        # merely been copied to a staging area, as in GROMACS builds through
        # at least 2022), the Python package metadata will not be available,
        # but through at least gmxapi 0.4, the `__version__` module attribute
        # is hard coded in the source files.
        gmxpy_version = parse(gmxapi.__version__)
    if gmxpy_version >= parse("0.4.0a3"):
        assert hasattr(core, "library_has_feature")
        assert "gmxapi_level" in config()
        assert parse(config()["gmxapi_level"]) >= parse("0.2")
        assert core.has_feature("create_context")
    assert not core.has_feature("spam")


@pytest.mark.skipif(
    not core.has_feature("mpi_bindings"),
    reason="Requires MPI bindings through mpi4py at package build time.",
)
@pytest.mark.withmpi_only
@pytest.mark.usefixtures("cleandir")
def test_mpi_bindings():
    from mpi4py import MPI

    rank_number = MPI.COMM_WORLD.Get_rank()
    comm_size = MPI.COMM_WORLD.Get_size()

    gmxapi_rank, gmxapi_size = core.mpi_report(MPI.COMM_WORLD)
    assert (gmxapi_rank, gmxapi_size) == (rank_number, comm_size)


@pytest.mark.skipif(
    not core.has_feature("mpi_bindings"),
    reason="Requires MPI bindings through mpi4py at package build time.",
)
@pytest.mark.withmpi_only
@pytest.mark.usefixtures("cleandir")
def test_mpi_sharing():
    from mpi4py import MPI

    rank_number = MPI.COMM_WORLD.Get_rank()

    # For non-MPI GROMACS, only offer a communicator of size 1.
    # We may be able to make resource allocation decisions based on larger
    # communicators in the future, but until then we should reject the
    # potentially wasteful resource assignment.
    color = 0
    if gmxapi.utility.config()["gmx_mpi_type"] != "library":
        if rank_number > 0:
            color = MPI.UNDEFINED
    sub_communicator = MPI.COMM_WORLD.Split(color, rank_number)
    if sub_communicator != MPI.COMM_NULL:
        try:
            api_context = core.create_context(sub_communicator)
            del api_context
        finally:
            sub_communicator.Free()

    # All GROMACS builds should be able to handle an ensemble of simulations
    # on disjoint sub-communicators of size 1.
    sub_communicator = MPI.COMM_WORLD.Split(rank_number, rank_number)
    if sub_communicator != MPI.COMM_NULL:
        try:
            api_context = core.create_context(sub_communicator)
            del api_context
        finally:
            sub_communicator.Free()

    # For MPI GROMACS, check that we can accept communicators of size > 1.
    if gmxapi.utility.config()["gmx_mpi_type"] == "library":
        ensemble_size = 2
        if rank_number < ensemble_size:
            color = 0
        else:
            color = MPI.UNDEFINED

        sub_communicator = MPI.COMM_WORLD.Split(color, rank_number)
        if sub_communicator != MPI.COMM_NULL:
            try:
                api_context = core.create_context(sub_communicator)
                del api_context
            finally:
                sub_communicator.Free()
