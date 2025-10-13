/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2011- The GROMACS Authors
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
 * \brief
 * Declares mock implementation of gmx::IAnalysisDataModule.
 *
 * Requires Google Mock.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \inlibraryapi
 * \ingroup module_analysisdata
 */
#ifndef GMX_ANALYSISDATA_TESTS_MOCK_DATAMODULE_H
#define GMX_ANALYSISDATA_TESTS_MOCK_DATAMODULE_H

#include <memory>

#include <gmock/gmock.h>

#include "gromacs/analysisdata/dataframe.h"
#include "gromacs/analysisdata/datamodule.h"
#include "gromacs/analysisdata/paralleloptions.h"

namespace gmx
{
namespace test
{

class AnalysisDataTestInput;
class TestReferenceChecker;

class MockAnalysisDataModule : public IAnalysisDataModule
{
public:
    explicit MockAnalysisDataModule(int flags);
    ~MockAnalysisDataModule() override;

    int flags() const override;

    MOCK_METHOD2(parallelDataStarted,
                 bool(AbstractAnalysisData* data, const AnalysisDataParallelOptions& options));
    MOCK_METHOD1(dataStarted, void(AbstractAnalysisData* data));
    MOCK_METHOD1(frameStarted, void(const AnalysisDataFrameHeader& header));
    MOCK_METHOD1(pointsAdded, void(const AnalysisDataPointSetRef& points));
    MOCK_METHOD1(frameFinished, void(const AnalysisDataFrameHeader& header));
    MOCK_METHOD1(frameFinishedSerial, void(int frameIndex));
    MOCK_METHOD0(dataFinished, void());

    void setupStaticCheck(const AnalysisDataTestInput& data, AbstractAnalysisData* source, bool bParallel);
    void setupStaticColumnCheck(const AnalysisDataTestInput& data, int firstcol, int n, AbstractAnalysisData* source);
    void setupStaticStorageCheck(const AnalysisDataTestInput& data,
                                 int                          storageCount,
                                 AbstractAnalysisData*        source);
    void setupReferenceCheck(const TestReferenceChecker& checker, AbstractAnalysisData* source);

private:
    class Impl;

    std::unique_ptr<Impl> impl_;
};

//! Smart pointer to manage an MockAnalysisDataModule object.
typedef std::shared_ptr<MockAnalysisDataModule> MockAnalysisDataModulePointer;

} // namespace test
} // namespace gmx

#endif
