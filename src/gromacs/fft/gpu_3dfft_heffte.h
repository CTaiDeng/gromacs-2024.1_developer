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
 *  \brief Declares the GPU 3D FFT routines.
 *  \author Gaurav Garg <gaugarg@nvidia.com>
 *  \ingroup module_fft
 */

#ifndef GMX_FFT_GPU_3DFFT_HEFFTE_H
#define GMX_FFT_GPU_3DFFT_HEFFTE_H

#include <heffte.h>

#include <memory>

#include "gromacs/fft/fft.h"
#include "gromacs/gpu_utils/device_stream.h"
#include "gromacs/gpu_utils/devicebuffer_datatype.h"
#include "gromacs/gpu_utils/gputraits.h"
#include "gromacs/gpu_utils/hostallocator.h"
#include "gromacs/utility/gmxmpi.h"

#include "gpu_3dfft_impl.h"

class DeviceContext;
class DeviceStream;

namespace gmx
{

/*! \internal \brief
 * A 3D FFT wrapper class for performing R2C/C2R transforms using heffte
 */
template<typename backend_tag>
class Gpu3dFft::ImplHeFfte : public Gpu3dFft::Impl
{
public:
    //! \copydoc Gpu3dFft::Impl::Impl
    ImplHeFfte(bool                 allocateRealGrid,
               MPI_Comm             comm,
               ArrayRef<const int>  gridSizesInXForEachRank,
               ArrayRef<const int>  gridSizesInYForEachRank,
               int                  nz,
               bool                 performOutOfPlaceFFT,
               const DeviceContext& context,
               const DeviceStream&  pmeStream,
               ivec                 realGridSize,
               ivec                 realGridSizePadded,
               ivec                 complexGridSizePadded,
               DeviceBuffer<float>* realGrid,
               DeviceBuffer<float>* complexGrid);

    /*! \brief Destroys the FFT plans. */
    ~ImplHeFfte() override;

    /*! \brief Performs the FFT transform in given direction
     *
     * \param[in]  dir           FFT transform direction specifier
     * \param[out] timingEvent   pointer to the timing event where timing data is recorded
     */
    void perform3dFft(gmx_fft_direction dir, CommandEvent* timingEvent) override;

private:
    heffte::gpu::vector<std::complex<float>> workspace_;

    std::unique_ptr<heffte::fft3d_r2c<backend_tag, int>> fftPlan_;

#if GMX_GPU_CUDA
    //! Raw stream for PME operations
    cudaStream_t pmeRawStream_;
    //! Local real grid
    heffte::gpu::vector<float> localRealGrid_;
    //! Local complex grid
    heffte::gpu::vector<std::complex<float>> localComplexGrid_;
#elif GMX_GPU_SYCL
    /*! \brief Raw stream for PME operations
     *
     * HeFFTe takes a std::reference_wrapper of the sycl::queue,
     * so we cannot pass the temporary value returned by
     * DeviceStream::stream().  Instead we keep a local copy to
     * pass to HeFFTe. */
    sycl::queue pmeRawStream_;
    //! Local real grid
    DeviceBuffer<float> localRealGrid_;
    //! Local complex grid
    DeviceBuffer<float> localComplexGrid_;
#endif
};

} // namespace gmx

#endif
