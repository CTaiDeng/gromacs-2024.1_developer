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

/*!\internal
 * \file
 * \brief
 * Implements gmx::OutputAdapterContainer.
 *
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 * \ingroup module_coordinateio
 */

#include "gmxpre.h"

#include "outputadaptercontainer.h"

#include <algorithm>

#include "gromacs/utility/exceptions.h"

namespace gmx
{

void OutputAdapterContainer::addAdapter(OutputAdapterPointer adapter, CoordinateFileFlags type)
{
    if (outputAdapters_[type] != nullptr)
    {
        GMX_THROW(InternalError("Trying to add adapter that has already been added"));
    }
    adapter->checkAbilityDependencies(abilities_);
    outputAdapters_[type] = std::move(adapter);
}

bool OutputAdapterContainer::isEmpty() const
{
    return std::none_of(outputAdapters_.begin(), outputAdapters_.end(), [](const auto& adapter) {
        return adapter != nullptr;
    });
}
} // namespace gmx
