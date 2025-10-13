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
 * Implements the ForceBuffers class
 *
 * \author Berk Hess <hess@kth.se>
 * \ingroup module_mdtypes
 */

#include "gmxpre.h"

#include "forcebuffers.h"

#include "gromacs/gpu_utils/hostallocator.h"

namespace gmx
{

ForceBuffers::ForceBuffers() :
    force_({}), forceMtsCombined_({}), view_({}, {}, false), useForceMtsCombined_(false)
{
}

ForceBuffers::ForceBuffers(const bool useForceMtsCombined, const PinningPolicy pinningPolicy) :
    force_({}, { pinningPolicy }),
    forceMtsCombined_({}),
    view_({}, {}, useForceMtsCombined),
    useForceMtsCombined_(useForceMtsCombined)
{
}

ForceBuffers::~ForceBuffers() = default;

ForceBuffers& ForceBuffers::operator=(ForceBuffers const& o)
{
    auto oForce = o.view().force();
    resize(oForce.size());
    std::copy(oForce.begin(), oForce.end(), view().force().begin());

    return *this;
}

PinningPolicy ForceBuffers::pinningPolicy() const
{
    return force_.get_allocator().pinningPolicy();
}

void ForceBuffers::resize(int numAtoms)
{
    force_.resizeWithPadding(numAtoms);
    if (useForceMtsCombined_)
    {
        forceMtsCombined_.resizeWithPadding(numAtoms);
    }
    view_ = ForceBuffersView(
            force_.arrayRefWithPadding(), forceMtsCombined_.arrayRefWithPadding(), useForceMtsCombined_);
}

} // namespace gmx
