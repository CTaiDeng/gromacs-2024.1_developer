/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2018- The GROMACS Authors
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

#include "gmxpre.h"

#include "gromacs/restraint/manager.h"

#include <gtest/gtest.h>

namespace
{

class DummyRestraint : public gmx::IRestraintPotential
{
public:
    ~DummyRestraint() override = default;

    gmx::PotentialPointData evaluate(gmx::Vector gmx_unused r1,
                                     gmx::Vector gmx_unused r2,
                                     double gmx_unused      t) override
    {
        return {};
    }

    void update(gmx::Vector gmx_unused v, gmx::Vector gmx_unused v0, double gmx_unused t) override
    {
    }

    std::vector<int> sites() const override { return std::vector<int>(); }

    void bindSession(gmxapi::SessionResources* session) override { (void)session; }
};

TEST(RestraintManager, restraintList)
{
    auto managerInstance = gmx::RestraintManager();
    managerInstance.addToSpec(std::make_shared<DummyRestraint>(), "a");
    managerInstance.addToSpec(std::make_shared<DummyRestraint>(), "b");
    EXPECT_EQ(managerInstance.countRestraints(), 2);
    managerInstance.clear();
    EXPECT_EQ(managerInstance.countRestraints(), 0);
    managerInstance.addToSpec(std::make_shared<DummyRestraint>(), "c");
    managerInstance.addToSpec(std::make_shared<DummyRestraint>(), "d");
    EXPECT_EQ(managerInstance.countRestraints(), 2);
}

} // end namespace
