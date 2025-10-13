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
#include "transformations.h"

#include "nblib/util/util.hpp"

namespace nblib
{

void sortInteractions(ListedInteractionData& interactions)
{
    auto sortOneElement = [](auto& interactionElement) {
        using InteractionContainerType = std::decay_t<decltype(interactionElement)>;
        using InteractionType          = typename InteractionContainerType::type;

        std::sort(begin(interactionElement.indices),
                  end(interactionElement.indices),
                  interactionSortKey<InteractionType>);
    };

    for_each_tuple(sortOneElement, interactions);
}

} // namespace nblib
