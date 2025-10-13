/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2021- The GROMACS Authors
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
 * \brief Defines the a helper struct managing reference temperature changes
 *
 * \author Pascal Merz <pascal.merz@me.com>
 * \ingroup module_modularsimulator
 */

#include "gmxpre.h"

#include "referencetemperaturemanager.h"

#include "gromacs/mdtypes/group.h"
#include "gromacs/utility/gmxassert.h"

namespace gmx
{

ReferenceTemperatureManager::ReferenceTemperatureManager(gmx_ekindata_t* ekindata) :
    ekindata_(ekindata)
{
    GMX_RELEASE_ASSERT(ekindata, "Need a valid ekindata object");
}

void ReferenceTemperatureManager::registerUpdateCallback(ReferenceTemperatureCallback referenceTemperatureCallback)
{
    callbacks_.emplace_back(std::move(referenceTemperatureCallback));
}

void ReferenceTemperatureManager::setReferenceTemperature(ArrayRef<const real> newReferenceTemperatures,
                                                          ReferenceTemperatureChangeAlgorithm algorithm)
{
    GMX_RELEASE_ASSERT(newReferenceTemperatures.ssize() == ekindata_->numTemperatureCouplingGroups(),
                       "Expected one new reference temperature per temperature group.");

    for (gmx::Index i = 0; i < gmx::ssize(newReferenceTemperatures); i++)
    {
        ekindata_->setCurrentReferenceTemperature(i, newReferenceTemperatures[i]);
    }
    for (const auto& callback : callbacks_)
    {
        callback(newReferenceTemperatures, algorithm);
    }
}

} // namespace gmx
