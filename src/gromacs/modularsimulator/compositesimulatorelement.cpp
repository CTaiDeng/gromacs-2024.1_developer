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

/*! \internal \file
 * \brief Defines the composite element for the modular simulator
 *
 * \author Pascal Merz <pascal.merz@me.com>
 * \ingroup module_modularsimulator
 */

#include "gmxpre.h"

#include "compositesimulatorelement.h"

#include "gromacs/mdlib/stat.h"

namespace gmx
{
CompositeSimulatorElement::CompositeSimulatorElement(
        std::vector<compat::not_null<ISimulatorElement*>>    elementCallList,
        std::vector<std::unique_ptr<gmx::ISimulatorElement>> elements,
        int                                                  frequency) :
    elementCallList_(std::move(elementCallList)),
    elementOwnershipList_(std::move(elements)),
    frequency_(frequency)
{
}

void CompositeSimulatorElement::scheduleTask(Step step, Time time, const RegisterRunFunction& registerRunFunction)
{
    if (do_per_step(step, frequency_))
    {
        for (auto& element : elementCallList_)
        {
            element->scheduleTask(step, time, registerRunFunction);
        }
    }
}

void CompositeSimulatorElement::elementSetup()
{
    for (auto& element : elementOwnershipList_)
    {
        element->elementSetup();
    }
}

void CompositeSimulatorElement::elementTeardown()
{
    for (auto& element : elementOwnershipList_)
    {
        element->elementTeardown();
    }
}

} // namespace gmx
