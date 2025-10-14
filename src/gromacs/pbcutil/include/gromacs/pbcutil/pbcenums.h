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

/*! \file
 * \brief
 * Defines enum classes for centering and unit cell types.
 *
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 * \libinternal
 * \ingroup module_pbcutil
 */
#ifndef GMX_PBCUTIL_PBCENUMS_H
#define GMX_PBCUTIL_PBCENUMS_H

namespace gmx
{

/*! \brief
 * Helper enum class to define centering types.
 */
enum class CenteringType : int
{
    Triclinic,
    Rectangular,
    Zero,
    Count
};

/*! \brief
 * Helper enum class to define Unit cell representation types.
 */
enum class UnitCellType : int
{
    Triclinic,
    Rectangular,
    Compact,
    Count
};

/*! \brief
 * Get names for the different centering types.
 *
 * \param[in] type What name needs to be provided.
 */
const char* centerTypeNames(CenteringType type);

/*! \brief
 * Get names for the different unit cell representation types.
 *
 * \param[in] type What name needs to be provided.
 */
const char* unitCellTypeNames(UnitCellType type);

} // namespace gmx

#endif
