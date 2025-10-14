/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2019- The GROMACS Authors
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
 *
 * \brief Declares functions that write timestamps to e.g. log files.
 *
 * \ingroup module_mdrunutility
 * \inlibraryapi
 */
#ifndef GMX_MDRUNUTILITY_PRINTTIME_H
#define GMX_MDRUNUTILITY_PRINTTIME_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>

struct gmx_walltime_accounting;
struct t_commrec;
struct t_inputrec;

//! Print time to \c out.
void print_time(FILE*                    out,
                gmx_walltime_accounting* walltime_accounting,
                int64_t                  step,
                const t_inputrec*        ir,
                const t_commrec*         cr);

/*! \brief Print date, time, MPI rank and a description of this point
 * in time.
 *
 * \param[in] log       logfile, or NULL to suppress output
 * \param[in] rank      MPI rank to include in the output
 * \param[in] title     Description to include in the output
 * \param[in] the_time  Seconds since the epoch, e.g. as reported by gmx_gettime
 */
void print_date_and_time(FILE* log, int rank, const char* title, double the_time);

//! Print start time to \c fplog.
void print_start(FILE* fplog, const t_commrec* cr, gmx_walltime_accounting* walltime_accounting, const char* name);

#endif
