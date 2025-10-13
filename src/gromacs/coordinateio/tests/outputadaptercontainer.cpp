/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2019- The GROMACS Authors
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

/*!\internal
 * \file
 * \brief
 * Tests for outputadaptercontainer.
 *
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 * \ingroup module_coordinateio
 */


#include "gmxpre.h"

#include "gromacs/coordinateio/outputadaptercontainer.h"

#include <algorithm>
#include <memory>

#include <gtest/gtest.h>

#include "gromacs/utility/exceptions.h"

#include "testmodule.h"

namespace gmx
{

namespace test
{

TEST(OutputAdapterContainer, MakeEmpty)
{
    OutputAdapterContainer container(CoordinateFileFlags::Base);
    EXPECT_TRUE(container.isEmpty());
}

TEST(OutputAdapterContainer, AddAdapter)
{
    OutputAdapterContainer container(CoordinateFileFlags::Base);
    container.addAdapter(std::make_unique<DummyOutputModule>(CoordinateFileFlags::Base),
                         CoordinateFileFlags::RequireNewFrameStartTime);
    EXPECT_FALSE(container.isEmpty());
}

TEST(OutputAdapterContainer, RejectBadAdapter)
{
    OutputAdapterContainer container(CoordinateFileFlags::Base);
    EXPECT_THROW(container.addAdapter(
                         std::make_unique<DummyOutputModule>(CoordinateFileFlags::RequireVelocityOutput),
                         CoordinateFileFlags::RequireVelocityOutput),
                 InconsistentInputError);
    EXPECT_TRUE(container.isEmpty());
}

TEST(OutputAdapterContainer, RejectDuplicateAdapter)
{
    OutputAdapterContainer container(CoordinateFileFlags::Base);
    EXPECT_NO_THROW(container.addAdapter(std::make_unique<DummyOutputModule>(CoordinateFileFlags::Base),
                                         CoordinateFileFlags::RequireNewFrameStartTime));
    EXPECT_FALSE(container.isEmpty());
    EXPECT_THROW(container.addAdapter(std::make_unique<DummyOutputModule>(CoordinateFileFlags::Base),
                                      CoordinateFileFlags::RequireNewFrameStartTime),
                 InternalError);
}

TEST(OutputAdapterContainer, AcceptMultipleAdapters)
{
    OutputAdapterContainer container(CoordinateFileFlags::Base);
    EXPECT_NO_THROW(container.addAdapter(std::make_unique<DummyOutputModule>(CoordinateFileFlags::Base),
                                         CoordinateFileFlags::RequireForceOutput));
    EXPECT_FALSE(container.isEmpty());
    EXPECT_NO_THROW(container.addAdapter(std::make_unique<DummyOutputModule>(CoordinateFileFlags::Base),
                                         CoordinateFileFlags::RequireVelocityOutput));
    EXPECT_FALSE(container.isEmpty());
}

} // namespace test

} // namespace gmx
