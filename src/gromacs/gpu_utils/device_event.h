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

/*! \libinternal \file
 *  \brief Declares DeviceEvent for all build configuraitons
 *
 *  This header may be included from any build configuration and
 *  defers valid GPU declarations to headers valid only in such
 *  build configurations.
 *
 *  \author Mark Abraham <mark.j.abraham@gmail.com>
 * \inlibraryapi
 */
#ifndef GMX_GPU_UTILS_DEVICE_EVENT_H
#define GMX_GPU_UTILS_DEVICE_EVENT_H

#include "config.h"

#include "gromacs/utility/exceptions.h"

#if !GMX_GPU || defined(DOXYGEN)

class DeviceStream;

// [[noreturn]] attributes must be added to the methods, so it's
// easier to silence the warning here and avoid them appearing in
// the Doxygen
#    ifdef __clang__
#        pragma clang diagnostic push
#        pragma clang diagnostic ignored "-Wmissing-noreturn"
#    endif

class DeviceEvent
{
public:
    DeviceEvent() = default;
    // Disable copy, move, and assignment. Move can be allowed, but not needed yet.
    DeviceEvent& operator=(const DeviceEvent&) = delete;
    DeviceEvent(const DeviceEvent&)            = delete;
    DeviceEvent& operator=(DeviceEvent&&) = delete;
    DeviceEvent(DeviceEvent&&)            = delete;

    /*! \brief Marks the synchronization point in the \p stream.
     * Should be followed by waitForEvent().
     */
    inline void mark(const DeviceStream& /*deviceStream*/) // NOLINT readability-convert-member-functions-to-static
    {
        GMX_THROW(gmx::NotImplementedError("Not implemented for non-GPU build"));
    }
    //! Synchronizes the host thread on the marked event.
    inline void wait() // NOLINT readability-convert-member-functions-to-static
    {
        GMX_THROW(gmx::NotImplementedError("Not implemented for non-GPU build"));
    }
    //! Checks the completion of the underlying event.
    inline bool isReady() // NOLINT readability-convert-member-functions-to-static
    {
        GMX_THROW(gmx::NotImplementedError("Not implemented for non-GPU build"));
    }
    //! Enqueues a wait for the recorded event in stream \p stream
    // NOLINTNEXTLINE readability-convert-member-functions-to-static
    inline void enqueueWait(const DeviceStream& /*deviceStream*/)
    {
        GMX_THROW(gmx::NotImplementedError("Not implemented for non-GPU build"));
    }
    //! Checks whether the underlying event was marked.
    inline bool isMarked() const // NOLINT readability-convert-member-functions-to-static
    {
        GMX_THROW(gmx::NotImplementedError("Not implemented for non-GPU build"));
    }

    //! Reset the event (not needed in CUDA)
    // NOLINTNEXTLINE readability-convert-member-functions-to-static
    inline void reset() // NOLINT readability-convert-member-functions-to-static
    {
        GMX_THROW(gmx::NotImplementedError("Not implemented for non-GPU build"));
    }
};

#    ifdef __clang__
#        pragma clang diagnostic pop
#    endif

#elif GMX_GPU_CUDA
#    include "device_event.cuh"
#elif GMX_GPU_OPENCL
#    include "device_event_ocl.h"
#elif GMX_GPU_SYCL
#    include "device_event_sycl.h"
#endif

#endif // GMX_GPU_UTILS_DEVICE_EVENT_H
