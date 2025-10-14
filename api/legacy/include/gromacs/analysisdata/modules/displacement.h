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

/*! \file
 * \brief
 * Declares gmx::AnalysisDataDisplacementModule.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \inpublicapi
 * \ingroup module_analysisdata
 */
#ifndef GMX_ANALYSISDATA_MODULES_DISPLACEMENT_H
#define GMX_ANALYSISDATA_MODULES_DISPLACEMENT_H

#include "gromacs/analysisdata/abstractdata.h"
#include "gromacs/analysisdata/datamodule.h"
#include "gromacs/utility/real.h"

namespace gmx
{

class AnalysisDataBinAverageModule;

/*! \brief
 * Data module for calculating displacements.
 *
 * Output data contains a frame for each frame in the input data except the
 * first one.  For each frame, there can be multiple points, each of which
 * describes displacement for a certain time difference ending that that frame.
 * The first column contains the time difference (backwards from the current
 * frame), and the remaining columns the sizes of the displacements.
 *
 * Current implementation is not very generic, but should be easy to extend.
 *
 * \inpublicapi
 * \ingroup module_analysisdata
 */
class AnalysisDataDisplacementModule : public AbstractAnalysisData, public AnalysisDataModuleSerial
{
public:
    AnalysisDataDisplacementModule();
    ~AnalysisDataDisplacementModule() override;

    /*! \brief
     * Sets the largest displacement time to be calculated.
     */
    void setMaxTime(real tmax);
    /*! \brief
     * Sets an histogram module that will receive a MSD histogram.
     *
     * If this function is not called, no histogram is calculated.
     */
    void setMSDHistogram(const std::shared_ptr<AnalysisDataBinAverageModule>& histm);

    int flags() const override;

    void dataStarted(AbstractAnalysisData* data) override;
    void frameStarted(const AnalysisDataFrameHeader& header) override;
    void pointsAdded(const AnalysisDataPointSetRef& points) override;
    void frameFinished(const AnalysisDataFrameHeader& header) override;
    void dataFinished() override;

private:
    AnalysisDataFrameRef tryGetDataFrameInternal(int index) const override;
    bool                 requestStorageInternal(int nframes) override;

    class Impl;

    std::unique_ptr<Impl> _impl;
};

//! Smart pointer to manage an AnalysisDataDisplacementModule object.
typedef std::shared_ptr<AnalysisDataDisplacementModule> AnalysisDataDisplacementModulePointer;

} // namespace gmx

#endif
