/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2014- The GROMACS Authors
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

#ifndef GMX_SIMD_IMPLEMENTATION_IBM_VSX_H
#define GMX_SIMD_IMPLEMENTATION_IBM_VSX_H

// At high optimization levels, gcc 7.2 gives false
// positives.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-parameter"

// While we do our best to also test VSX with Power7, that depends on having
// access to big-endian hardware, so for the long term our focus will be
// little-endian Power8.

#include "impl_ibm_vsx_definitions.h"
#include "impl_ibm_vsx_general.h"
#include "impl_ibm_vsx_simd4_float.h"
#include "impl_ibm_vsx_simd_double.h"
#include "impl_ibm_vsx_simd_float.h"
#include "impl_ibm_vsx_util_double.h"
#include "impl_ibm_vsx_util_float.h"

#pragma GCC diagnostic pop

#endif // GMX_SIMD_IMPLEMENTATION_IBM_VSX_H
