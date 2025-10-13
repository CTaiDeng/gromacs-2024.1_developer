/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2015- The GROMACS Authors
 * Copyright (C) 2025- GaoZheng
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

/*! \libinternal\file
 * \brief
 * Low-level utility for converting 64 bit int to int (the
 * size of which is hardware dependent), printing
 * a warning if an overflow will occur.
 *
 * \inlibraryapi
 * \ingroup module_utility
 */
#ifndef GMX_UTILITY_INT64_TO_INT_H
#define GMX_UTILITY_INT64_TO_INT_H

#include "gromacs/utility/futil.h"

/*! \brief Convert a int64_t value to int.
 *
 * \param[in] step The step number (or other int64)
 * \param[in] warn If warn!=NULL a warning message will be written
 *                 to stderr when step does not fit in an int,
 *                 the first line is:
 *                 "WARNING during %s:", where warn is printed in %s.
 * \return the truncated step number.
 */
int int64_to_int(int64_t step, const char* warn);

#endif
