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
 * Implements nblib utilities for system setup
 *
 * \author Victor Holanda <victor.holanda@cscs.ch>
 * \author Joe Jordan <ejjordan@kth.se>
 * \author Prashanth Kanduri <kanduri@cscs.ch>
 * \author Sebastian Keller <keller@cscs.ch>
 * \author Artem Zhmurov <zhmurov@gmail.com>
 */

#ifndef NBLIB_UTIL_SETUP_H
#define NBLIB_UTIL_SETUP_H

#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "nblib/basicdefinitions.h"
#include "nblib/vector.h"

namespace gmx
{
template<typename T>
class ArrayRef;
} // namespace gmx

namespace nblib
{

/*! \brief Generate velocities from a Maxwell Boltzmann distribution
 *
 * masses should be the same as the ones specified for the Topology object
 */
std::vector<Vec3> generateVelocity(real Temperature, unsigned int seed, std::vector<real> const& masses);

//! \brief Check within the container of gmx::RVecs for a NaN or inf
bool isRealValued(gmx::ArrayRef<const Vec3> values);

//! \brief Zero a cartesian buffer
void zeroCartesianArray(gmx::ArrayRef<Vec3> cartesianArray);

} // namespace nblib

#endif // NBLIB_UTIL_SETUP_H
