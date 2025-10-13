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
 * \brief Implements PME halo exchange and PME-FFT grid conversion functions.
 *
 * \author Gaurav Garg <gaugarg@nvidia.com>
 *
 * \ingroup module_ewald
 */
#ifndef GMX_EWALD_PME_GPU_GRID_H
#define GMX_EWALD_PME_GPU_GRID_H

#include "gromacs/gpu_utils/devicebuffer_datatype.h"

struct PmeGpu;
struct gmx_wallcycle;
typedef struct gmx_parallel_3dfft* gmx_parallel_3dfft_t;

/*! \libinternal \brief
 * Grid Halo exchange after PME spread
 * ToDo: Current implementation transfers halo region from/to only immediate neighbours
 * And, expects that overlapSize <= local grid width.
 * Implement exchange with multiple neighbors to remove this limitation
 * ToDo: Current implementation synchronizes pmeStream to make sure data is ready on GPU after
 * spread. Consider using events for this synchnozation.
 *
 * \param[in]  pmeGpu                 The PME GPU structure.
 * \param[in]  wcycle                 The wallclock counter.
 */
void pmeGpuGridHaloExchange(const PmeGpu* pmeGpu, gmx_wallcycle* wcycle);

/*! \libinternal \brief
 * Grid reverse Halo exchange before PME gather
 * ToDo: Current implementation transfers halo region from/to only immediate neighbours
 * And, expects that overlapSize <= local grid width.
 * Implement exchange with multiple neighbors to remove this limitation
 * ToDo: Current implementation synchronizes pmeStream to make sure data is ready on GPU after FFT
 * to PME grid conversion. Consider using events for this synchnozation.
 *
 * \param[in]  pmeGpu                 The PME GPU structure.
 * \param[in]  wcycle                 The wallclock counter.
 */
void pmeGpuGridHaloExchangeReverse(const PmeGpu* pmeGpu, gmx_wallcycle* wcycle);

/*! \libinternal \brief
 * Copy PME Grid with overlap region to host FFT grid and vice-versa. Used in mixed mode PME decomposition
 *
 * \param[in]  pmeGpu                 The PME GPU structure.
 * \param[in]  h_fftRealGrid          FFT grid on host
 * \param[in]  fftSetup               Host FFT setup structure
 * \param[in]  gridIndex              Grid index which is to be converted
 *
 * \tparam  pmeToFft                  A boolean which tells if this is conversion from PME grid to FFT grid or reverse
 */
template<bool pmetofft>
void convertPmeGridToFftGrid(const PmeGpu*         pmeGpu,
                             float*                h_fftRealGrid,
                             gmx_parallel_3dfft_t* fftSetup,
                             int                   gridIndex);

/*! \libinternal \brief
 * Copy PME Grid with overlap region to device FFT grid and vice-versa. Used in full GPU PME decomposition
 *
 * \param[in]  pmeGpu                 The PME GPU structure.
 * \param[in]  d_fftRealGrid          FFT grid on device
 * \param[in]  gridIndex              Grid index which is to be converted
 *
 * \tparam  pmeToFft                  A boolean which tells if this is conversion from PME grid to FFT grid or reverse
 */
template<bool pmetofft>
void convertPmeGridToFftGrid(const PmeGpu* pmeGpu, DeviceBuffer<float>* d_fftRealGrid, int gridIndex);

extern template void convertPmeGridToFftGrid<true>(const PmeGpu* /*pmeGpu*/,
                                                   float* /*h_fftRealGrid*/,
                                                   gmx_parallel_3dfft_t* /*fftSetup*/,
                                                   int /*gridIndex*/);

extern template void convertPmeGridToFftGrid<false>(const PmeGpu* /*pmeGpu*/,
                                                    float* /*h_fftRealGrid*/,
                                                    gmx_parallel_3dfft_t* /*fftSetup*/,
                                                    int /*gridIndex*/);

extern template void convertPmeGridToFftGrid<true>(const PmeGpu* /*pmeGpu*/,
                                                   DeviceBuffer<float>* /*d_fftRealGrid*/,
                                                   int /*gridIndex*/);

extern template void convertPmeGridToFftGrid<false>(const PmeGpu* /*pmeGpu*/,
                                                    DeviceBuffer<float>* /*d_fftRealGrid*/,
                                                    int /*gridIndex*/);

#endif
