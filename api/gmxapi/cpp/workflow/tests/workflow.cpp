/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2018- The GROMACS Authors
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

#include "workflow.h"

#include <memory>

#include "testingconfiguration.h"
#include "workflow_impl.h"

namespace gmxapi
{

namespace testing
{

namespace
{

//! Create a work spec, then the implementation graph, then the container
TEST_F(GmxApiTest, BuildApiWorkflowImpl)
{
    makeTprFile(100);
    // Create work spec
    auto node = std::make_unique<gmxapi::MDNodeSpecification>(runner_.tprFileName_);
    EXPECT_NE(node, nullptr);

    // Create key
    std::string key{ "MD" };
    key.append(runner_.tprFileName_);

    // Create graph (workflow implementation object)
    gmxapi::Workflow::Impl impl;
    impl[key] = std::move(node);
    EXPECT_EQ(impl.count(key), 1);
    EXPECT_EQ(impl.size(), 1);

    // Create workflow container
    EXPECT_NO_THROW(gmxapi::Workflow work{ std::move(impl) });
}

//! Create from create() method(s)
TEST_F(GmxApiTest, CreateApiWorkflow)
{
    makeTprFile(100);
    auto work = gmxapi::Workflow::create(runner_.tprFileName_);
    EXPECT_NE(work, nullptr);
}

} // end anonymous namespace

} // end namespace testing

} // end namespace gmxapi
