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

set(GMXPY_TEST_FILES
    test_commandline.py
    test_exceptions.py
    test_mdrun.py
    test_operation.py
    )

# Make sure pytest is found or produce a fatal error
if(NOT GMXAPI_PYTEST_FOUND)
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import pytest; print(pytest.__version__)"
                    RESULT_VARIABLE pytest_not_found OUTPUT_VARIABLE pytest_version ERROR_QUIET)
    if(pytest_not_found)
        message(FATAL_ERROR "Running the tests requires pytest. Please install it manually"
                " (try: ${PYTHON_EXECUTABLE} -m pip install pytest)")
    elseif(pytest_version VERSION_LESS 3.0)
        message(FATAL_ERROR "Running the tests requires pytest >= 3.0. Found: ${pytest_version}"
                "Please update it (try: ${PYTHON_EXECUTABLE} -m pip install -U pytest)")
    endif()
    set(GMXAPI_PYTEST_FOUND TRUE CACHE INTERNAL "Suppress checking for Python pytest module.")
endif()

add_custom_target(gmxapi_pytest
                  COMMAND ${PYTHON_EXECUTABLE} -m pytest
                    -rA -l --log-cli-level ERROR ${CMAKE_CURRENT_SOURCE_DIR}
                  DEPENDS _gmxapi
                  WORKING_DIRECTORY ${GMXAPI_PYTHON_STAGING_DIR})
# The current test fixtures require the `gmx` tool-wrapper executable.
add_dependencies(gmxapi_pytest gmx)

add_dependencies(check gmxapi_pytest)

if(MPIEXEC_EXECUTABLE)
    if(NOT GMXAPI_MPI4PY_FOUND)
        if (MPI_C_COMPILER)
            set(_mpicc "MPICC=${MPI_C_COMPILER} ")
        endif ()

        execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import mpi4py; print(mpi4py.__version__)"
                        RESULT_VARIABLE mpi4py_not_found OUTPUT_VARIABLE mpi4py_version ERROR_QUIET)
        if(mpi4py_not_found)
            message(STATUS "Running the MPI-enable tests requires mpi4py. Please install it manually. "
                    "(Try: `${_mpicc}${PYTHON_EXECUTABLE} -m pip install mpi4py`, "
                    "then clear the GMXAPI_MPI4PY_FOUND cache variable "
                    "or remove the CMakeCache.txt, and configure again.)")
        elseif(mpi4py_version VERSION_LESS 3.0.3)
            message(STATUS "Running the MPI-enabled tests requires mpi4py >= 3.0.3. Found: ${mpi4py_version}"
                    "Please update it (try: ${_mpicc}${PYTHON_EXECUTABLE} -m pip install -U mpi4py)")
        endif()

        unset(_mpicc)
        set(GMXAPI_MPI4PY_FOUND TRUE CACHE INTERNAL "Suppress checking for Python mpi4py module.")
    endif()

    # In case this output becomes too noisy or confusing, Note:
    #    For Multiple Instruction Multiple Data Model for OpenMPI mpirun, we can
    #    make the non-root rank less noisy by providing different argv after a colon (`:`)
    #    (https://www.open-mpi.org/doc/v3.0/man1/mpiexec.1.php)
    add_custom_target(gmxapi_pytest_mpi
                      COMMAND ${MPIEXEC_EXECUTABLE} -n 4
                        ${PYTHON_EXECUTABLE} -m mpi4py -m pytest -x -rA -l --log-cli-level ERROR
                        ${CMAKE_CURRENT_SOURCE_DIR}
                      DEPENDS _gmxapi
                      WORKING_DIRECTORY ${GMXAPI_PYTHON_STAGING_DIR})
    # The current test fixtures require the `gmx` tool-wrapper executable.
    add_dependencies(gmxapi_pytest_mpi gmx)

    add_dependencies(check gmxapi_pytest_mpi)
endif()
