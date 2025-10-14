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
 *  \brief Implements GPU 3D FFT routines for SYCL.
 *
 *  \author Andrey Alekseenko <al42and@gmail.com>
 *  \author Mark Abraham <mark.j.abraham@gmail.com>
 *  \ingroup module_fft
 */

#include "gmxpre.h"

#include "gpu_3dfft_sycl.h"

#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/exceptions.h"

namespace gmx
{

// [[noreturn]] attributes must be added in the common headers, so it's easier to silence the warning here
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-noreturn"

Gpu3dFft::ImplSycl::ImplSycl(bool /*allocateRealGrid*/,
                             MPI_Comm /*comm*/,
                             ArrayRef<const int> /*gridSizesInXForEachRank*/,
                             ArrayRef<const int> /*gridSizesInYForEachRank*/,
                             const int /*nz*/,
                             bool /*performOutOfPlaceFFT*/,
                             const DeviceContext& /*context*/,
                             const DeviceStream& /*pmeStream*/,
                             ivec /*realGridSize*/,
                             ivec /*realGridSizePadded*/,
                             ivec /*complexGridSizePadded*/,
                             DeviceBuffer<float>* /*realGrid*/,
                             DeviceBuffer<float>* /*complexGrid*/)
{
    GMX_THROW(NotImplementedError("Using SYCL build without GPU 3DFFT support"));
}

Gpu3dFft::ImplSycl::~ImplSycl() = default;

void Gpu3dFft::ImplSycl::perform3dFft(gmx_fft_direction /*dir*/, CommandEvent* /*timingEvent*/)
{
    GMX_THROW(NotImplementedError("Using SYCL build without GPU 3DFFT support"));
}

#pragma clang diagnostic pop

} // namespace gmx
