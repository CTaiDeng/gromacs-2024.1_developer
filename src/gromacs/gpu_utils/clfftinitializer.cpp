/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2018- The GROMACS Authors
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
 * Implements ClfftInitializer class.
 *
 * \author Aleksei Iupinov <a.yupinov@gmail.com>
 */

#include "gmxpre.h"

#include "clfftinitializer.h"

#include "config.h"

#include <mutex>

#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/stringutil.h"

#if GMX_GPU_OPENCL && GMX_GPU_FFT_CLFFT
#    include <clFFT.h>
#endif

namespace gmx
{

namespace
{

#if GMX_GPU_OPENCL && GMX_GPU_FFT_CLFFT
/*! \brief The clFFT library may only be initialized once per process,
 * and this is orchestrated by this shared value and mutex.
 *
 * This ensures that thread-MPI and OpenMP builds can't accidentally
 * initialize it more than once. */
//! @{
bool       g_clfftInitialized = false;
std::mutex g_clfftMutex;
//! @}
#endif

} // namespace

ClfftInitializer::ClfftInitializer()
{
#if GMX_GPU_OPENCL && GMX_GPU_FFT_CLFFT
    std::lock_guard<std::mutex> guard(g_clfftMutex);
    clfftSetupData              fftSetup;
    int                         initErrorCode = clfftInitSetupData(&fftSetup);
    if (initErrorCode != 0)
    {
        GMX_THROW(InternalError(formatString(
                "Failed to initialize the clFFT library, error code %d", initErrorCode)));
    }
    initErrorCode = clfftSetup(&fftSetup);
    if (initErrorCode != 0)
    {
        GMX_THROW(InternalError(formatString(
                "Failed to initialize the clFFT library, error code %d", initErrorCode)));
    }
    g_clfftInitialized = true;
#endif
}

ClfftInitializer::~ClfftInitializer()
{
#if GMX_GPU_OPENCL && GMX_GPU_FFT_CLFFT
    std::lock_guard<std::mutex> guard(g_clfftMutex);
    if (g_clfftInitialized)
    {
        clfftTeardown();
        // TODO: log non-zero return values (errors)
    }
    g_clfftInitialized = false;
#endif
}

} // namespace gmx
