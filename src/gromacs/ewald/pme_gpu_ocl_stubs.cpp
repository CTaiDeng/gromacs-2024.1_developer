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
 * \brief Implements stubs of high-level PME GPU functions for OpenCL.
 *
 * \author Gaurav Garg <gaugarg@nvidia.com>
 *
 * \ingroup module_ewald
 */
#include "gmxpre.h"

#include "gromacs/fft/parallel_3dfft.h"

#include "pme_gpu_grid.h"
#include "pme_gpu_types.h"
#include "pme_gpu_types_host.h"
#include "pme_gpu_types_host_impl.h"

// [[noreturn]] attributes must be added in the common headers, so it's easier to silence the warning here
#if defined(__clang__)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wmissing-noreturn"
#endif // (__clang__)

void pmeGpuGridHaloExchange(const PmeGpu* /*pmeGpu*/, gmx_wallcycle* /*wcycle*/)
{
    GMX_THROW(gmx::NotImplementedError("PME decomposition is not implemented in OpenCL"));
}

void pmeGpuGridHaloExchangeReverse(const PmeGpu* /*pmeGpu*/, gmx_wallcycle* /*wcycle*/)
{
    GMX_THROW(gmx::NotImplementedError("PME decomposition is not implemented in OpenCL"));
}

template<bool forward>
void convertPmeGridToFftGrid(const PmeGpu* /*pmeGpu*/,
                             float* /*h_fftRealGrid*/,
                             gmx_parallel_3dfft_t* /*fftSetup*/,
                             const int /*gridIndex*/)
{
    GMX_THROW(gmx::NotImplementedError("PME decomposition is not implemented in OpenCL"));
}

template<bool forward>
void convertPmeGridToFftGrid(const PmeGpu* /*pmeGpu*/, DeviceBuffer<float>* /*d_fftRealGrid*/, const int /*gridIndex*/)
{
    GMX_THROW(gmx::NotImplementedError("PME decomposition is not implemented in OpenCL"));
}

template void convertPmeGridToFftGrid<true>(const PmeGpu* /*pmeGpu*/,
                                            float* /*h_fftRealGrid*/,
                                            gmx_parallel_3dfft_t* /*fftSetup*/,
                                            const int /*gridIndex*/);

template void convertPmeGridToFftGrid<false>(const PmeGpu* /*pmeGpu*/,
                                             float* /*h_fftRealGrid*/,
                                             gmx_parallel_3dfft_t* /*fftSetup*/,
                                             const int /*gridIndex*/);

template void convertPmeGridToFftGrid<true>(const PmeGpu* /*pmeGpu*/,
                                            DeviceBuffer<float>* /*d_fftRealGrid*/,
                                            const int /*gridIndex*/);

template void convertPmeGridToFftGrid<false>(const PmeGpu* /*pmeGpu*/,
                                             DeviceBuffer<float>* /*d_fftRealGrid*/,
                                             const int /*gridIndex*/);

#if defined(__clang__)
#    pragma clang diagnostic pop
#endif // (__clang__)
