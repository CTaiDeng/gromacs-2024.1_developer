/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2018- The GROMACS Authors
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

/*! \libinternal \file
 * \brief Provides utility functions for MiMiC QM/MM
 * \inlibraryapi
 *
 * \author Viacheslav Bolnykh <v.bolnykh@hpc-leap.eu>
 * \ingroup module_mimic
 */
#ifndef GMX_MIMIC_MIMICUTILS_H
#define GMX_MIMIC_MIMICUTILS_H

#include <vector>

struct gmx_mtop_t;

/*! \brief Generates the list of QM atoms
 *
 * This function generates vector containing global IDs of QM atoms
 * based on the information stored in the QM/MM group (egcQMMM)
 *
 * \param[in]    mtop   Global topology object
 * \return              The list of global IDs of QM atoms
 */
std::vector<int> genQmmmIndices(const gmx_mtop_t& mtop);

#endif // GMX_MIMIC_MIMICUTILS_H
