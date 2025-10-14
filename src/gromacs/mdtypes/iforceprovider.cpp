/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2017- The GROMACS Authors
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
 * Implements classes from iforceprovider.h.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_mdtypes
 */
#include "gmxpre.h"

#include "iforceprovider.h"

#include <vector>

#include "gromacs/utility/arrayref.h"

using namespace gmx;

class ForceProviders::Impl
{
public:
    std::vector<IForceProvider*> providers_;
};

ForceProviders::ForceProviders() : impl_(new Impl) {}

ForceProviders::~ForceProviders() {}

void ForceProviders::addForceProvider(gmx::IForceProvider* provider)
{
    impl_->providers_.push_back(provider);
}

bool ForceProviders::hasForceProvider() const
{
    return !impl_->providers_.empty();
}

void ForceProviders::calculateForces(const ForceProviderInput& forceProviderInput,
                                     ForceProviderOutput*      forceProviderOutput) const
{
    for (auto* provider : impl_->providers_)
    {
        provider->calculateForces(forceProviderInput, forceProviderOutput);
    }
}
