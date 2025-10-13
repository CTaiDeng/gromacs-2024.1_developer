/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2013- The GROMACS Authors
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

/*! \file
 * \brief
 * Declares gmx::AnalysisDataLifetimeModule.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \inpublicapi
 * \ingroup module_analysisdata
 */
#ifndef GMX_ANALYSISDATA_MODULES_LIFETIME_H
#define GMX_ANALYSISDATA_MODULES_LIFETIME_H

#include <memory>

#include "gromacs/analysisdata/arraydata.h"
#include "gromacs/analysisdata/datamodule.h"

namespace gmx
{

/*! \brief
 * Data module for computing lifetime histograms for columns in input data.
 *
 * The input data set is treated as a boolean array: each value that is present
 * (AnalysisDataValue::isPresent() returns true) and is >0 is treated as
 * present, other values are treated as absent.
 * For each input data set, analyzes the columns to identify the intervals
 * where a column is continuously present.
 * Produces a histogram from the lengths of these intervals.
 * Input data should have frames with evenly spaced x values.
 *
 * Output data contains one column for each data set in the input data.
 * This column gives the lifetime histogram for the corresponding data set.
 * x axis in the output is spaced the same as in the input data, and extends
 * as long as required to cover all the histograms.
 * Histograms are padded with zeros as required to be of the same length.
 * setCumulative() can be used to alter the handling of subintervals in the
 * output histogram.
 *
 * The output data becomes available only after the input data has been
 * finished.
 *
 * \inpublicapi
 * \ingroup module_analysisdata
 */
class AnalysisDataLifetimeModule : public AbstractAnalysisArrayData, public AnalysisDataModuleSerial
{
public:
    AnalysisDataLifetimeModule();
    ~AnalysisDataLifetimeModule() override;

    /*! \brief
     * Sets a cumulative histogram mode.
     *
     * \param[in] bCumulative If true, all subintervals of a long
     *   interval are also explicitly added into the histogram.
     *
     * Does not throw.
     */
    void setCumulative(bool bCumulative);

    int flags() const override;

    void dataStarted(AbstractAnalysisData* data) override;
    void frameStarted(const AnalysisDataFrameHeader& header) override;
    void pointsAdded(const AnalysisDataPointSetRef& points) override;
    void frameFinished(const AnalysisDataFrameHeader& header) override;
    void dataFinished() override;

private:
    class Impl;

    std::unique_ptr<Impl> impl_;
};

//! Smart pointer to manage an AnalysisDataLifetimeModule object.
typedef std::shared_ptr<AnalysisDataLifetimeModule> AnalysisDataLifetimeModulePointer;

} // namespace gmx

#endif
