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

#ifndef GMX_HARDWARE_PRINTHARDWARE_H
#define GMX_HARDWARE_PRINTHARDWARE_H

#include <cstdio>

struct gmx_hw_info_t;

namespace gmx
{
class MDLogger;
}

/* Print information about the detected hardware to fplog (if != NULL)
 * and to stderr on the main rank of the main simulation.
 */
void gmx_print_detected_hardware(FILE*                fplog,
                                 bool                 warnToStdErr,
                                 const gmx::MDLogger& mdlog,
                                 const gmx_hw_info_t* hwinfo);

#endif
