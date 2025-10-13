/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2017- The GROMACS Authors
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
 * \brief Defines a routine to check the number of AVX512 fma units
 *
 * \author Erik Lindahl <erik.lindahl@gmail.com>
 * \inlibraryapi
 * \ingroup module_hardware
 */

namespace gmx
{

/*! \brief Test whether machine has dual AVX512 FMA units
 *
 * \return 1 or 2 for the number of AVX512 FMA units if AVX512
 *         support is present, 0 if we know the hardware does
 *         not have AVX512 support, or -1 if the test cannot
 *         run because the compiler lacked AVX512 support.
 */
int identifyAvx512FmaUnits();

} // namespace gmx
