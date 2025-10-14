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
 *
 * \brief
 * Declares function for compute transformation coordinate values and forces
 *
 * \author Oliver Fleetwood <oliver.fleetwood@gmail.com>
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 * \author Joe Jordan <ejjordan@kth.se>
 * \author Berk Hess <hess@kth.se>
 *
 */
#ifndef GMX_PULL_TRANSFORMATIONCOORDINATE_H
#define GMX_PULL_TRANSFORMATIONCOORDINATE_H

struct pull_t;
struct pull_coord_work_t;

namespace gmx
{
template<typename>
class ArrayRef;

/*! \brief Calculates pull->coord[coord_ind].spatialData.value for a transformation pull coordinate
 *
 * This requires the values of the pull coordinates of lower indices to be set
 * \param[in] coord  The (transformation) coordinate to compute the value for
 * \param[in] variableCoords  Pull coordinates used as variables, entries 0 to coord->coordIndex
 *                            will be used
 * \param[in] t               The time
 *
 * \returns Transformation value for pull coordinate.
 *
 * \throws InconsistentInputError when the expression uses coordinates with index equal to or larger
 *         than the index of \p coord.
 */
double getTransformationPullCoordinateValue(pull_coord_work_t*                coord,
                                            ArrayRef<const pull_coord_work_t> variableCoords,
                                            double                            t);

/*! \brief Distributes the force on a transformation pull coordinates to the involved coordinates of lower rank
 *
 * Note: There is no recursion applied here, which means that this function needs to be called
 *       in a reverse loop over the pull coordinates to ensure that forces on recursive transformation
 *       coordinates are distributed correctly.
 *
 * \param[in,out] pcrd            The transformation pull coordinate to act on
 * \param[in,out] variableCoords  List of variable coords up to the coord index of \p pcrd
 */
void distributeTransformationPullCoordForce(pull_coord_work_t*               pcrd,
                                            gmx::ArrayRef<pull_coord_work_t> variableCoords);

} // namespace gmx

#endif // GMX_PULL_TRANSFORMATIONCOORDINATE_H
