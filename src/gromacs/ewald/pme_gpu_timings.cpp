/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2016- The GROMACS Authors
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
 *  \brief Implements PME GPU timing events wrappers.
 *
 *  \author Aleksei Iupinov <a.yupinov@gmail.com>
 * \ingroup module_ewald
 */

#include "gmxpre.h"

#include "pme_gpu_timings.h"

#include "gromacs/utility/gmxassert.h"

#include "pme_gpu_internal.h"
#include "pme_gpu_types_host.h"
#include "pme_gpu_types_host_impl.h"

bool pme_gpu_timings_enabled(const PmeGpu* pmeGpu)
{
    return pmeGpu->archSpecific->useTiming;
}

void pme_gpu_start_timing(const PmeGpu* pmeGpu, PmeStage pmeStageId)
{
    if (pme_gpu_timings_enabled(pmeGpu))
    {
        GMX_ASSERT(pmeStageId < PmeStage::Count, "Wrong PME GPU timing event index");
        pmeGpu->archSpecific->timingEvents[pmeStageId].openTimingRegion(pmeGpu->archSpecific->pmeStream_);
    }
}

void pme_gpu_stop_timing(const PmeGpu* pmeGpu, PmeStage pmeStageId)
{
    if (pme_gpu_timings_enabled(pmeGpu))
    {
        GMX_ASSERT(pmeStageId < PmeStage::Count, "Wrong PME GPU timing event index");
        pmeGpu->archSpecific->timingEvents[pmeStageId].closeTimingRegion(pmeGpu->archSpecific->pmeStream_);
    }
}

void pme_gpu_get_timings(const PmeGpu* pmeGpu, gmx_wallclock_gpu_pme_t* timings)
{
    if (pme_gpu_timings_enabled(pmeGpu))
    {
        GMX_RELEASE_ASSERT(timings, "Null GPU timing pointer");
        for (auto key : keysOf(timings->timing))
        {
            timings->timing[key].t = pmeGpu->archSpecific->timingEvents[key].getTotalTime();
            timings->timing[key].c = pmeGpu->archSpecific->timingEvents[key].getCallCount();
        }
    }
}

void pme_gpu_update_timings(const PmeGpu* pmeGpu)
{
    if (pme_gpu_timings_enabled(pmeGpu))
    {
        for (const auto& activeTimer : pmeGpu->archSpecific->activeTimers)
        {
            pmeGpu->archSpecific->timingEvents[activeTimer].getLastRangeTime();
        }
    }
}

void pme_gpu_reinit_timings(const PmeGpu* pmeGpu)
{
    if (pme_gpu_timings_enabled(pmeGpu))
    {
        pmeGpu->archSpecific->activeTimers.clear();
        pmeGpu->archSpecific->activeTimers.insert(PmeStage::SplineAndSpread);
        const auto& settings = pme_gpu_settings(pmeGpu);
        // TODO: no separate gtPME_SPLINE and gtPME_SPREAD as they are not used currently
        if (settings.performGPUFFT)
        {
            pmeGpu->archSpecific->activeTimers.insert(PmeStage::FftTransformC2R);
            pmeGpu->archSpecific->activeTimers.insert(PmeStage::FftTransformR2C);
        }
        if (settings.performGPUSolve)
        {
            pmeGpu->archSpecific->activeTimers.insert(PmeStage::Solve);
        }
        if (settings.performGPUGather)
        {
            pmeGpu->archSpecific->activeTimers.insert(PmeStage::Gather);
        }
    }
}

void pme_gpu_reset_timings(const PmeGpu* pmeGpu)
{
    if (pme_gpu_timings_enabled(pmeGpu))
    {
        for (auto key : keysOf(pmeGpu->archSpecific->timingEvents))
        {
            pmeGpu->archSpecific->timingEvents[key].reset();
        }
    }
}
