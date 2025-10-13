/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2020- The GROMACS Authors
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

/*! \inpublicapi \file
 * \brief
 * Implements nblib simulation box
 *
 * \author Victor Holanda <victor.holanda@cscs.ch>
 * \author Joe Jordan <ejjordan@kth.se>
 * \author Prashanth Kanduri <kanduri@cscs.ch>
 * \author Sebastian Keller <keller@cscs.ch>
 */
#ifndef NBLIB_BOX_H
#define NBLIB_BOX_H

#include "nblib/basicdefinitions.h"

namespace nblib
{

/*! \brief Box specifies the size of periodic simulation systems
 *
 * \inpublicapi
 * \ingroup nblib
 *
 * Currently only cubic and rectangular boxes are supported.
 *
 */
class Box final
{
public:
    using LegacyMatrix = matrix;

    //! Construct a cubic box.
    explicit Box(real l);

    //! Construct a rectangular box.
    Box(real x, real y, real z);

    //! Return the full matrix that specifies the box. Used for gromacs setup code.
    [[nodiscard]] LegacyMatrix const& legacyMatrix() const { return legacyMatrix_; }

private:
    //! \brief check two boxes for equality
    friend bool operator==(const Box& rhs, const Box& lhs);

    //! Stores data in the GROMACS legacy data type
    LegacyMatrix legacyMatrix_;
};

} // namespace nblib
#endif // NBLIB_BOX_H
