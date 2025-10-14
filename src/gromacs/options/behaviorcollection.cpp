/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2015- The GROMACS Authors
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
 * Implements gmx::OptionsBehaviorCollection.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_options
 */
#include "gmxpre.h"

#include "behaviorcollection.h"

#include "gromacs/options/ioptionsbehavior.h"

namespace gmx
{

IOptionsBehavior::~IOptionsBehavior() {}

OptionsBehaviorCollection::OptionsBehaviorCollection(Options* options) : options_(options) {}

OptionsBehaviorCollection::~OptionsBehaviorCollection() {}

void OptionsBehaviorCollection::addBehavior(const OptionsBehaviorPointer& behavior)
{
    behaviors_.reserve(behaviors_.size() + 1);
    behavior->initBehavior(options_);
    behaviors_.push_back(behavior);
}

void OptionsBehaviorCollection::optionsFinishing()
{
    for (const OptionsBehaviorPointer& behavior : behaviors_)
    {
        behavior->optionsFinishing(options_);
    }
}

void OptionsBehaviorCollection::optionsFinished()
{
    for (const OptionsBehaviorPointer& behavior : behaviors_)
    {
        behavior->optionsFinished();
    }
}

} // namespace gmx
