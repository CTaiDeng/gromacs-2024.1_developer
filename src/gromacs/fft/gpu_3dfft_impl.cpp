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
 *  \brief Implements stub GPU 3D FFT routines for CPU-only builds
 *
 *  \author Mark Abraham <mark.j.abraham@gmail.com>
 *  \author Gaurav Garg <gaugarg@nvidia.com>
 *  \ingroup module_fft
 */

#include "gmxpre.h"

#include "gpu_3dfft_impl.h"

#include "gromacs/gpu_utils/devicebuffer.h"

namespace gmx
{

Gpu3dFft::Impl::Impl() : complexGrid_(nullptr) {}

Gpu3dFft::Impl::Impl(bool performOutOfPlaceFFT) :
    performOutOfPlaceFFT_(performOutOfPlaceFFT), complexGrid_(nullptr)
{
}

void Gpu3dFft::Impl::allocateComplexGrid(const ivec           complexGridSizePadded,
                                         DeviceBuffer<float>* realGrid,
                                         DeviceBuffer<float>* complexGrid,
                                         const DeviceContext& context)
{
    if (performOutOfPlaceFFT_)
    {
        const int newComplexGridSize =
                complexGridSizePadded[XX] * complexGridSizePadded[YY] * complexGridSizePadded[ZZ] * 2;

        allocateDeviceBuffer(complexGrid, newComplexGridSize, context);
    }
    else
    {
        *complexGrid = *realGrid;
    }

    complexGrid_ = *complexGrid;
}

void Gpu3dFft::Impl::deallocateComplexGrid()
{
    if (performOutOfPlaceFFT_)
    {
        freeDeviceBuffer(&complexGrid_);
    }
}

Gpu3dFft::Impl::~Impl() = default;

} // namespace gmx
