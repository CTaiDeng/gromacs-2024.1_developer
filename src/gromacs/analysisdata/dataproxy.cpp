/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2010- The GROMACS Authors
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
 * Implements gmx::AnalysisDataProxy.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_analysisdata
 */
#include "gmxpre.h"

#include "dataproxy.h"

#include "gromacs/analysisdata/dataframe.h"
#include "gromacs/analysisdata/datamodulemanager.h"
#include "gromacs/utility/gmxassert.h"

namespace gmx
{

AnalysisDataProxy::AnalysisDataProxy(int firstColumn, int columnSpan, AbstractAnalysisData* data) :
    source_(*data), firstColumn_(firstColumn), columnSpan_(columnSpan), bParallel_(false)
{
    GMX_RELEASE_ASSERT(data != nullptr, "Source data must not be NULL");
    GMX_RELEASE_ASSERT(firstColumn >= 0 && columnSpan > 0, "Invalid proxy column");
    setMultipoint(source_.isMultipoint());
}


int AnalysisDataProxy::frameCount() const
{
    return source_.frameCount();
}


AnalysisDataFrameRef AnalysisDataProxy::tryGetDataFrameInternal(int index) const
{
    AnalysisDataFrameRef frame = source_.tryGetDataFrame(index);
    if (!frame.isValid())
    {
        return AnalysisDataFrameRef();
    }
    return AnalysisDataFrameRef(frame, firstColumn_, columnSpan_);
}


bool AnalysisDataProxy::requestStorageInternal(int nframes)
{
    return source_.requestStorage(nframes);
}


int AnalysisDataProxy::flags() const
{
    return efAllowMultipoint | efAllowMulticolumn | efAllowMissing | efAllowMultipleDataSets;
}


void AnalysisDataProxy::dataStarted(AbstractAnalysisData* data)
{
    GMX_RELEASE_ASSERT(data == &source_, "Source data mismatch");
    setDataSetCount(data->dataSetCount());
    for (int i = 0; i < data->dataSetCount(); ++i)
    {
        setColumnCount(i, columnSpan_);
    }
    moduleManager().notifyDataStart(this);
}


bool AnalysisDataProxy::parallelDataStarted(AbstractAnalysisData*              data,
                                            const AnalysisDataParallelOptions& options)
{
    GMX_RELEASE_ASSERT(data == &source_, "Source data mismatch");
    setDataSetCount(data->dataSetCount());
    for (int i = 0; i < data->dataSetCount(); ++i)
    {
        setColumnCount(i, columnSpan_);
    }
    moduleManager().notifyParallelDataStart(this, options);
    bParallel_ = !moduleManager().hasSerialModules();
    return bParallel_;
}


void AnalysisDataProxy::frameStarted(const AnalysisDataFrameHeader& frame)
{
    if (bParallel_)
    {
        moduleManager().notifyParallelFrameStart(frame);
    }
    else
    {
        moduleManager().notifyFrameStart(frame);
    }
}


void AnalysisDataProxy::pointsAdded(const AnalysisDataPointSetRef& points)
{
    AnalysisDataPointSetRef columns(points, firstColumn_, columnSpan_);
    if (columns.columnCount() > 0)
    {
        if (bParallel_)
        {
            moduleManager().notifyParallelPointsAdd(columns);
        }
        else
        {
            moduleManager().notifyPointsAdd(columns);
        }
    }
}


void AnalysisDataProxy::frameFinished(const AnalysisDataFrameHeader& header)
{
    if (bParallel_)
    {
        moduleManager().notifyParallelFrameFinish(header);
    }
    else
    {
        moduleManager().notifyFrameFinish(header);
    }
}

void AnalysisDataProxy::frameFinishedSerial(int frameIndex)
{
    if (bParallel_)
    {
        // The x and dx values are unused in this case.
        AnalysisDataFrameHeader header(frameIndex, 0.0, 0.0);
        moduleManager().notifyFrameFinish(header);
    }
}


void AnalysisDataProxy::dataFinished()
{
    moduleManager().notifyDataFinish();
}

} // namespace gmx
