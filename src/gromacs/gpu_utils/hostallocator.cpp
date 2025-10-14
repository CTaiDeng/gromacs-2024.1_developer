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
 * \brief Implements gmx::HostAllocationPolicy for allocating memory
 * suitable for e.g. GPU transfers on CUDA.
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 */
#include "gmxpre.h"

#include "hostallocator.h"

#include <cstddef>

#include <memory>

#include "gromacs/gpu_utils/gpu_utils.h"
#include "gromacs/gpu_utils/pmalloc.h"
#include "gromacs/utility/alignedallocator.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/stringutil.h"

namespace gmx
{

HostAllocationPolicy::HostAllocationPolicy(PinningPolicy pinningPolicy) :
    pinningPolicy_(pinningPolicy)
{
}

std::size_t HostAllocationPolicy::alignment() const noexcept
{
    return (pinningPolicy_ == PinningPolicy::PinnedIfSupported ? PageAlignedAllocationPolicy::alignment()
                                                               : AlignedAllocationPolicy::alignment());
}

void* HostAllocationPolicy::malloc(std::size_t bytes) const noexcept
{
    if (pinningPolicy_ == PinningPolicy::PinnedIfSupported)
    {
        void* p;
        pmalloc(&p, bytes);
        return p;
    }
    else
    {
        return AlignedAllocationPolicy::malloc(bytes);
    }
}

void HostAllocationPolicy::free(void* buffer) const noexcept
{
    if (buffer == nullptr)
    {
        // Nothing to do
        return;
    }
    if (pinningPolicy_ == PinningPolicy::PinnedIfSupported)
    {
        pfree(buffer);
    }
    else
    {
        AlignedAllocationPolicy::free(buffer);
    }
}

} // namespace gmx
