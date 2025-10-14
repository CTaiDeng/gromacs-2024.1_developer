/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2021- The GROMACS Authors
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

/*! \internal \file
 * \brief
 * Implements functionality to compute virials from force data
 *
 * \author Victor Holanda <victor.holanda@cscs.ch>
 * \author Joe Jordan <ejjordan@kth.se>
 * \author Prashanth Kanduri <kanduri@cscs.ch>
 * \author Sebastian Keller <keller@cscs.ch>
 */
#ifndef NBLIB_VIRIALS_H
#define NBLIB_VIRIALS_H

#include "nblib/basicdefinitions.h"
#include "nblib/vector.h"

namespace gmx
{
template<typename>
class ArrayRef;
}

namespace nblib
{
class Box;

//! Computes virial tensor and stores it in an array of size 9
void computeVirialTensor(gmx::ArrayRef<const Vec3> coordinates,
                         gmx::ArrayRef<const Vec3> forces,
                         gmx::ArrayRef<const Vec3> shiftVectors,
                         gmx::ArrayRef<const Vec3> shiftForces,
                         const Box&                box,
                         gmx::ArrayRef<real>       virialOutput);

} // namespace nblib
#endif // NBLIB_VIRIALS_H
