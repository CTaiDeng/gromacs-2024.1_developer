/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2013- The GROMACS Authors
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
 * Implements gmx::AnalysisDataFrameAverager.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_analysisdata
 */
#include "gmxpre.h"

#include "frameaverager.h"

#include "gromacs/analysisdata/dataframe.h"
#include "gromacs/utility/gmxassert.h"

namespace gmx
{

void AnalysisDataFrameAverager::setColumnCount(int columnCount)
{
    GMX_RELEASE_ASSERT(columnCount >= 0, "Invalid column count");
    GMX_RELEASE_ASSERT(values_.empty(), "Cannot initialize multiple times");
    values_.resize(columnCount);
}

void AnalysisDataFrameAverager::addValue(int index, real value)
{
    AverageItem& item  = values_[index];
    const double delta = value - item.average;
    item.samples += 1;
    item.average += delta / item.samples;
    item.squaredSum += delta * (value - item.average);
}

void AnalysisDataFrameAverager::addPoints(const AnalysisDataPointSetRef& points)
{
    const int firstColumn = points.firstColumn();
    GMX_ASSERT(static_cast<size_t>(firstColumn + points.columnCount()) <= values_.size(),
               "Initialized with too few columns");
    for (int i = 0; i < points.columnCount(); ++i)
    {
        if (points.present(i))
        {
            addValue(firstColumn + i, points.y(i));
        }
    }
}

void AnalysisDataFrameAverager::finish()
{
    bFinished_ = true;
}

} // namespace gmx
