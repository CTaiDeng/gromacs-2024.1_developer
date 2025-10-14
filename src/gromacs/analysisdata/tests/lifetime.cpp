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
 * Tests for functionality of analysis data lifetime module.
 *
 * These tests check that gmx::AnalysisDataLifetimeModule computes lifetimes
 * correctly with simple input data.
 * Checking is done using gmx::test::AnalysisDataTestFixture and reference
 * data.  Also the input data is written to the reference data to catch
 * out-of-date reference.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_analysisdata
 */
#include "gmxpre.h"

#include "gromacs/analysisdata/modules/lifetime.h"

#include <gtest/gtest.h>

#include "gromacs/analysisdata/analysisdata.h"
#include "gromacs/analysisdata/tests/datatest.h"

#include "testutils/testasserts.h"

using gmx::test::AnalysisDataTestInput;

namespace
{

// Simple input data for gmx::AnalysisDataLifetimeModule tests.
class SimpleInputData
{
public:
    static const AnalysisDataTestInput& get()
    {
        static SimpleInputData singleton;
        return singleton.data_;
    }

    SimpleInputData() : data_(1, false)
    {
        data_.setColumnCount(0, 3);
        data_.addFrameWithValues(1.0, 1.0, 1.0, 1.0);
        data_.addFrameWithValues(2.0, 1.0, 0.0, 1.0);
        data_.addFrameWithValues(3.0, 0.0, 1.0, 1.0);
    }

private:
    AnalysisDataTestInput data_;
};

// Input data with multiple data sets for gmx::AnalysisDataLifetimeModule tests.
class MultiDataSetInputData
{
public:
    static const AnalysisDataTestInput& get()
    {
        static MultiDataSetInputData singleton;
        return singleton.data_;
    }

    MultiDataSetInputData() : data_(2, false)
    {
        using gmx::test::AnalysisDataTestInputFrame;
        data_.setColumnCount(0, 2);
        data_.setColumnCount(1, 2);
        AnalysisDataTestInputFrame& frame1 = data_.addFrame(1.0);
        frame1.addPointSetWithValues(0, 0, 1.0, 1.0);
        frame1.addPointSetWithValues(1, 0, 0.0, 0.0);
        AnalysisDataTestInputFrame& frame2 = data_.addFrame(2.0);
        frame2.addPointSetWithValues(0, 0, 1.0, 0.0);
        frame2.addPointSetWithValues(1, 0, 1.0, 0.0);
        AnalysisDataTestInputFrame& frame3 = data_.addFrame(3.0);
        frame3.addPointSetWithValues(0, 0, 1.0, 0.0);
        frame3.addPointSetWithValues(1, 0, 1.0, 1.0);
    }

private:
    AnalysisDataTestInput data_;
};


/********************************************************************
 * Tests for gmx::AnalysisDataLifetimeModule.
 */

//! Test fixture for gmx::AnalysisDataLifetimeModule.
typedef gmx::test::AnalysisDataTestFixture LifetimeModuleTest;

TEST_F(LifetimeModuleTest, BasicTest)
{
    const AnalysisDataTestInput& input = SimpleInputData::get();
    gmx::AnalysisData            data;
    ASSERT_NO_THROW_GMX(setupDataObject(input, &data));

    gmx::AnalysisDataLifetimeModulePointer module(new gmx::AnalysisDataLifetimeModule);
    module->setCumulative(false);
    data.addModule(module);

    ASSERT_NO_THROW_GMX(addStaticCheckerModule(input, &data));
    ASSERT_NO_THROW_GMX(addReferenceCheckerModule("InputData", &data));
    ASSERT_NO_THROW_GMX(addReferenceCheckerModule("Lifetime", module.get()));
    ASSERT_NO_THROW_GMX(presentAllData(input, &data));
}

TEST_F(LifetimeModuleTest, CumulativeTest)
{
    const AnalysisDataTestInput& input = SimpleInputData::get();
    gmx::AnalysisData            data;
    ASSERT_NO_THROW_GMX(setupDataObject(input, &data));

    gmx::AnalysisDataLifetimeModulePointer module(new gmx::AnalysisDataLifetimeModule);
    module->setCumulative(true);
    data.addModule(module);

    ASSERT_NO_THROW_GMX(addStaticCheckerModule(input, &data));
    ASSERT_NO_THROW_GMX(addReferenceCheckerModule("InputData", &data));
    ASSERT_NO_THROW_GMX(addReferenceCheckerModule("Lifetime", module.get()));
    ASSERT_NO_THROW_GMX(presentAllData(input, &data));
}

TEST_F(LifetimeModuleTest, HandlesMultipleDataSets)
{
    const AnalysisDataTestInput& input = MultiDataSetInputData::get();
    gmx::AnalysisData            data;
    ASSERT_NO_THROW_GMX(setupDataObject(input, &data));

    gmx::AnalysisDataLifetimeModulePointer module(new gmx::AnalysisDataLifetimeModule);
    module->setCumulative(false);
    data.addModule(module);

    ASSERT_NO_THROW_GMX(addStaticCheckerModule(input, &data));
    ASSERT_NO_THROW_GMX(addReferenceCheckerModule("InputData", &data));
    ASSERT_NO_THROW_GMX(addReferenceCheckerModule("Lifetime", module.get()));
    ASSERT_NO_THROW_GMX(presentAllData(input, &data));
}

} // namespace
