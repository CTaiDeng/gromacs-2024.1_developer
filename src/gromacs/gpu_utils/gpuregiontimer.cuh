/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2016- The GROMACS Authors
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

/*! \libinternal \file
 *  \brief Implements the GPU region timer for CUDA.
 *
 *  \author Aleksei Iupinov <a.yupinov@gmail.com>
 *
 *  \inlibraryapi
 */

#ifndef GMX_GPU_UTILS_GPUREGIONTIMER_CUH
#define GMX_GPU_UTILS_GPUREGIONTIMER_CUH

#include "gromacs/gpu_utils/cudautils.cuh"
#include "gromacs/gpu_utils/gputraits.cuh"

#include "gpuregiontimer.h"

/*! \libinternal \brief
 * This is a GPU region timing implementation for CUDA.
 * It provides methods for measuring the last timespan.
 * Copying/assignment is disabled since the underlying timing events are owned by this.
 */
class GpuRegionTimerImpl
{
    //! The underlying timing event pair - the beginning and the end of the timespan
    cudaEvent_t eventStart_, eventStop_;

public:
    GpuRegionTimerImpl()
    {
        const int eventFlags = cudaEventDefault;
        CU_RET_ERR(cudaEventCreate(&eventStart_, eventFlags), "GPU timing creation failure");
        CU_RET_ERR(cudaEventCreate(&eventStop_, eventFlags), "GPU timing creation failure");
    }
    ~GpuRegionTimerImpl()
    {
        CU_RET_ERR(cudaEventDestroy(eventStart_), "GPU timing destruction failure");
        CU_RET_ERR(cudaEventDestroy(eventStop_), "GPU timing destruction failure");
    }
    //! No copying
    GpuRegionTimerImpl(const GpuRegionTimerImpl&) = delete;
    //! No assignment
    GpuRegionTimerImpl& operator=(GpuRegionTimerImpl&&) = delete;
    //! Moving is disabled but can be considered in the future if needed
    GpuRegionTimerImpl(GpuRegionTimerImpl&&) = delete;

    /*! \brief Will be called before the region start. */
    inline void openTimingRegion(const DeviceStream& deviceStream)
    {
        CU_RET_ERR(cudaEventRecord(eventStart_, deviceStream.stream()),
                   "GPU timing recording failure");
    }

    /*! \brief Will be called after the region end. */
    inline void closeTimingRegion(const DeviceStream& deviceStream)
    {
        CU_RET_ERR(cudaEventRecord(eventStop_, deviceStream.stream()),
                   "GPU timing recording failure");
    }

    /*! \brief Returns the last measured region timespan (in milliseconds) and calls reset() */
    inline double getLastRangeTime()
    {
        float milliseconds = 0.0;
        CU_RET_ERR(cudaEventElapsedTime(&milliseconds, eventStart_, eventStop_),
                   "GPU timing update failure");
        reset();
        return milliseconds;
    }

    /*! \brief Resets internal state */
    inline void reset() {}

    /*! \brief Returns a new raw timing event
     * for passing into individual GPU API calls.
     * This is just a dummy in CUDA.
     */
    static inline CommandEvent* fetchNextEvent() { return nullptr; }
};

//! Short-hand for external use
using GpuRegionTimer = GpuRegionTimerWrapper<GpuRegionTimerImpl>;

#endif
