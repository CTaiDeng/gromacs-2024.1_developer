/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2020- The GROMACS Authors
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
 *
 * Implement transformations and manipulations of ListedInteractionData,
 * such as sorting and splitting
 *
 * \author Victor Holanda <victor.holanda@cscs.ch>
 * \author Joe Jordan <ejjordan@kth.se>
 * \author Prashanth Kanduri <kanduri@cscs.ch>
 * \author Sebastian Keller <keller@cscs.ch>
 * \author Artem Zhmurov <zhmurov@gmail.com>
 */
#ifndef NBLIB_LISTEDFORCES_TRANSFORMATIONS_H
#define NBLIB_LISTEDFORCES_TRANSFORMATIONS_H

#include <algorithm>

#include "nblib/listed_forces/definitions.h"

namespace nblib
{

namespace detail
{

/*! \brief tuple ordering for two center interactions
 *
 * \param t input array
 * \return  ordered array
 */
inline std::array<int, 2> nblibOrdering(const std::array<int, 2>& t)
{
    // for bonds (two coordinate indices),
    // we choose to store the lower sequence ID first. this allows for better unit tests
    // that are agnostic to how the input was set up
    int id1 = std::min(std::get<0>(t), std::get<1>(t));
    int id2 = std::max(std::get<0>(t), std::get<1>(t));

    return std::array<int, 2>{ id1, id2 };
}

/*! \brief tuple ordering for three center interactions
 *
 * \param t input array
 * \return  ordered array
 */
inline std::array<int, 3> nblibOrdering(const std::array<int, 3>& t)
{
    // for angles (three coordinate indices),
    // we choose to store the two non-center coordinate indices sorted.
    // such that ret[0] < ret[2] always (ret = returned tuple)
    int id1 = std::min(std::get<0>(t), std::get<2>(t));
    int id3 = std::max(std::get<0>(t), std::get<2>(t));

    return std::array<int, 3>{ id1, std::get<1>(t), id3 };
}

/*! \brief tuple ordering for four center interactions
 *
 * \param t input array
 * \return  ordered array
 */
inline std::array<int, 4> nblibOrdering(const std::array<int, 4>& t)
{
    return t;
}

/*! \brief tuple ordering for five center interactions
 *
 * \param t input array
 * \return  ordered array
 */
inline std::array<int, 5> nblibOrdering(const std::array<int, 5>& t)
{
    return t;
}

} // namespace detail

//! \brief sort key function object to sort 2-center interactions
template<class Interaction>
std::enable_if_t<Contains<Interaction, SupportedTwoCenterTypes>{}, bool>
interactionSortKey(const InteractionIndex<Interaction>& lhs, const InteractionIndex<Interaction>& rhs)
{
    return lhs < rhs;
}

//! \brief sort key function object to sort 3-center interactions
template<class Interaction>
std::enable_if_t<Contains<Interaction, SupportedThreeCenterTypes>{}, bool>
interactionSortKey(const InteractionIndex<Interaction>& lhs, const InteractionIndex<Interaction>& rhs)
{
    // position [1] is the center atom of the angle and is the only sort key
    // to allow use of std::equal_range to obtain a range of all angles with a given central atom
    return lhs[1] < rhs[1];
}

//! \brief sort key function object to sort 4-center interactions
template<class Interaction>
std::enable_if_t<Contains<Interaction, SupportedFourCenterTypes>{}, bool>
interactionSortKey(const InteractionIndex<Interaction>& lhs, const InteractionIndex<Interaction>& rhs)
{
    // we only take the first center-axis-particle into account
    // this allows use of std::equal_range to find all four-center interactions with a given j-index
    // Note that there exists a second version that compares the k-index which is used in
    // aggregate transformation
    return lhs[1] < rhs[1];
}

//! \brief sort key function object to sort 5-center interactions
//! \brief sort key function object to sort 4-center interactions
template<class Interaction>
std::enable_if_t<Contains<Interaction, SupportedFiveCenterTypes>{}, bool>
interactionSortKey(const InteractionIndex<Interaction>& lhs, const InteractionIndex<Interaction>& rhs)
{
    return lhs < rhs;
}

//! \brief sorts all interaction indices according to the keys defined in the implementation
void sortInteractions(ListedInteractionData& interactions);

} // namespace nblib
#endif // NBLIB_LISTEDFORCES_TRANSFORMATIONS_H
