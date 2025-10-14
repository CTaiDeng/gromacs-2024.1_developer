/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2012- The GROMACS Authors
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
 * Implements classes in dataframe.h.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_analysisdata
 */
#include "gmxpre.h"

#include "gromacs/analysisdata/dataframe.h"

#include "gromacs/utility/gmxassert.h"

namespace gmx
{

/********************************************************************
 * AnalysisDataFrameHeader
 */

AnalysisDataFrameHeader::AnalysisDataFrameHeader() : index_(-1), x_(0.0), dx_(0.0) {}


AnalysisDataFrameHeader::AnalysisDataFrameHeader(int index, real x, real dx) :
    index_(index), x_(x), dx_(dx)
{
    GMX_ASSERT(index >= 0, "Invalid frame index");
}


/********************************************************************
 * AnalysisDataPointSetRef
 */

AnalysisDataPointSetRef::AnalysisDataPointSetRef(const AnalysisDataFrameHeader&  header,
                                                 const AnalysisDataPointSetInfo& pointSetInfo,
                                                 const AnalysisDataValuesRef&    values) :
    header_(header),
    dataSetIndex_(pointSetInfo.dataSetIndex()),
    firstColumn_(pointSetInfo.firstColumn()),
    values_(constArrayRefFromArray(&*values.begin() + pointSetInfo.valueOffset(), pointSetInfo.valueCount()))
{
    GMX_ASSERT(header_.isValid(), "Invalid point set reference should not be constructed");
}


AnalysisDataPointSetRef::AnalysisDataPointSetRef(const AnalysisDataFrameHeader&        header,
                                                 const std::vector<AnalysisDataValue>& values) :
    header_(header), dataSetIndex_(0), firstColumn_(0), values_(values)
{
    GMX_ASSERT(header_.isValid(), "Invalid point set reference should not be constructed");
}


AnalysisDataPointSetRef::AnalysisDataPointSetRef(const AnalysisDataPointSetRef& points,
                                                 int                            firstColumn,
                                                 int                            columnCount) :
    header_(points.header()), dataSetIndex_(points.dataSetIndex()), firstColumn_(0)
{
    GMX_ASSERT(firstColumn >= 0, "Invalid first column");
    GMX_ASSERT(columnCount >= 0, "Invalid column count");
    if (points.lastColumn() < firstColumn || points.firstColumn() >= firstColumn + columnCount
        || columnCount == 0)
    {
        return;
    }
    AnalysisDataValuesRef::const_iterator begin        = points.values().begin();
    int                                   pointsOffset = firstColumn - points.firstColumn();
    if (pointsOffset > 0)
    {
        // Offset pointer if the first column is not the first in points.
        begin += pointsOffset;
    }
    else
    {
        // Take into account if first column is before the first in points.
        firstColumn_ = -pointsOffset;
        columnCount -= -pointsOffset;
    }
    // Decrease column count if there are not enough columns in points.
    AnalysisDataValuesRef::const_iterator end = begin + columnCount;
    if (pointsOffset + columnCount > points.columnCount())
    {
        end = points.values().end();
    }
    values_ = AnalysisDataValuesRef(begin, end);
}


bool AnalysisDataPointSetRef::allPresent() const
{
    AnalysisDataValuesRef::const_iterator i;
    for (i = values_.begin(); i != values_.end(); ++i)
    {
        if (!i->isPresent())
        {
            return false;
        }
    }
    return true;
}


/********************************************************************
 * AnalysisDataFrameRef
 */

AnalysisDataFrameRef::AnalysisDataFrameRef() {}


AnalysisDataFrameRef::AnalysisDataFrameRef(const AnalysisDataFrameHeader&      header,
                                           const AnalysisDataValuesRef&        values,
                                           const AnalysisDataPointSetInfosRef& pointSets) :
    header_(header), values_(values), pointSets_(pointSets)
{
    GMX_ASSERT(!pointSets_.empty(), "There must always be a point set");
}


AnalysisDataFrameRef::AnalysisDataFrameRef(const AnalysisDataFrameHeader&               header,
                                           const std::vector<AnalysisDataValue>&        values,
                                           const std::vector<AnalysisDataPointSetInfo>& pointSets) :
    header_(header), values_(values), pointSets_(pointSets)
{
    GMX_ASSERT(!pointSets_.empty(), "There must always be a point set");
}


AnalysisDataFrameRef::AnalysisDataFrameRef(const AnalysisDataFrameRef& frame, int firstColumn, int columnCount) :
    header_(frame.header()),
    values_(constArrayRefFromArray(&frame.values_[firstColumn], columnCount)),
    pointSets_(frame.pointSets_)
{
    // FIXME: This doesn't produce a valid internal state, although it does
    // work in some cases. The point sets cannot be correctly managed here, but
    // need to be handles by the data proxy class.
    GMX_ASSERT(firstColumn >= 0, "Invalid first column");
    GMX_ASSERT(columnCount >= 0, "Invalid column count");
    GMX_ASSERT(pointSets_.size() == 1U, "Subsets of frames only supported with simple data");
    GMX_ASSERT(firstColumn + columnCount <= ssize(values_), "Invalid last column");
}


bool AnalysisDataFrameRef::allPresent() const
{
    GMX_ASSERT(isValid(), "Invalid data frame accessed");
    AnalysisDataValuesRef::const_iterator i;
    for (i = values_.begin(); i != values_.end(); ++i)
    {
        if (!i->isPresent())
        {
            return false;
        }
    }
    return true;
}

} // namespace gmx
