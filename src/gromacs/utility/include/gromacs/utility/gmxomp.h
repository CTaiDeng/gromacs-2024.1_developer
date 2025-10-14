/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2012- The GROMACS Authors
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
 * Declares OpenMP wrappers to avoid conditional compilation.
 *
 * This module defines wrappers for OpenMP API functions and enables compiling
 * code without conditional compilation even when OpenMP is turned off in the
 * build system.
 * Therefore, OpenMP API functions should always be used through these wrappers
 * and omp.h should never be directly included.  Instead, this header should be
 * used whenever OpenMP API functions are needed.
 *
 * \inlibraryapi
 * \ingroup module_utility
 */
#ifndef GMX_UTILITY_OMP_H
#define GMX_UTILITY_OMP_H

/*! \addtogroup module_utility
 * \{
 */

/*! \brief
 * Returns an integer equal to or greater than the number of threads
 * that would be available if a parallel region without num_threads were
 * defined at that point in the code.
 *
 * Acts as a wrapper for omp_get_max_threads().
 */
int gmx_omp_get_max_threads();

/*! \brief
 * Returns the number of processors available when the function is called.
 *
 * Acts as a wrapper around omp_get_num_procs().
 */
int gmx_omp_get_num_procs();

/*! \brief
 * Returns the thread number of the thread executing within its thread team.
 *
 * Acts as a wrapper for omp_get_thread_num().
 */
int gmx_omp_get_thread_num();

/*! \brief
 * Sets the number of threads in subsequent parallel regions, unless overridden
 * by a num_threads clause.
 *
 * Acts as a wrapper for omp_set_num_threads().
 */
void gmx_omp_set_num_threads(int num_threads);

/*! \brief
 * Check for externally set thread affinity to avoid conflicts with \Gromacs
 * internal setting.
 *
 * \param[out] message  Receives the message to be shown to the user.
 * \returns `true` if we can set thread affinity ourselves.
 *
 * The KMP_AFFINITY environment variable is used by Intel, GOMP_CPU_AFFINITY
 * by the GNU compilers (Intel also honors it well).  If any of the variables
 * is set, we should honor it and disable the internal pinning.
 *
 * If this function returns `false`, the caller is responsible to disable the
 * pinning, show the message from \p *message to the user, and free the memory
 * allocated for \p *message.
 * If the return value is `true`, \p *message is NULL.
 */
bool gmx_omp_check_thread_affinity(char** message);

/*! \} */

#endif
