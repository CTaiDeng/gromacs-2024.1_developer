/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2013- The GROMACS Authors
 * Copyright (C) 2025 GaoZheng
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 *
 * ---
 *
 * This file is part of a modified version of the GROMACS molecular simulation package.
 * For details on the original project, consult https://www.gromacs.org.
 *
 * To help fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out https://www.gromacs.org.
 */

/*! \libinternal \file
 * \brief
 * Wraps mpi.h usage in Gromacs.
 *
 * This header wraps the MPI header <mpi.h>, and should be included instead of
 * that one.  It abstracts away the case that depending on compilation
 * settings, MPI routines may be provided by <mpi.h> or by thread-MPI.
 * In the absence of MPI, this header still declares some types for
 * convenience.  It also disables MPI C++ bindings that can cause compilation
 * issues.
 *
 * \inlibraryapi
 * \ingroup module_utility
 */
#ifndef GMX_UTILITY_GMXMPI_H
#define GMX_UTILITY_GMXMPI_H

#include "config.h"

/*! \cond */
#if GMX_LIB_MPI
/* MPI C++ binding is deprecated and can cause name conflicts (e.g. stdio/mpi seek) */
#    define MPICH_SKIP_MPICXX 1
#    define OMPI_SKIP_MPICXX 1
/* disable bindings for SGI MPT also */
#    define MPI_NO_CPPBIND 1
#    include <mpi.h>
/* Starting with 2.2 MPI_INT64_T is required. Earlier version still might have it.
   In theory MPI_Datatype doesn't have to be a #define, but current available MPI
   implementations (OpenMPI + MPICH (+derivates)) use #define and future versions
   should support 2.2. */
#    if (MPI_VERSION == 1 || (MPI_VERSION == 2 && MPI_SUBVERSION < 2)) && !defined MPI_INT64_T
#        include <limits.h>
#        if LONG_MAX == 9223372036854775807L
#            define MPI_INT64_T MPI_LONG
#        elif LONG_LONG_MAX == 9223372036854775807L
#            define MPI_INT64_T MPI_LONG_LONG
#        else
#            error No MPI_INT64_T and no 64 bit integer found.
#        endif
#    endif /*MPI_INT64_T*/
#else
#    if GMX_THREAD_MPI
#        include "thread_mpi/mpi_bindings.h" /* IWYU pragma: export */
#        include "thread_mpi/tmpi.h"         /* IWYU pragma: export */
#    else
typedef void* MPI_Comm;
typedef void* MPI_Request;
typedef void* MPI_Status;
typedef void* MPI_Group;
#        define MPI_COMM_NULL nullptr
#        define MPI_GROUP_NULL nullptr
#        define MPI_COMM_WORLD nullptr
#    endif
#endif
//! \endcond

#endif
