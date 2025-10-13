/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2015- The GROMACS Authors
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

/*! \internal \file
 * \brief
 * Tests for gmx::CpuInfo
 *
 * \author Erik Lindahl <erik.lindahl@gmail.com>
 * \ingroup module_hardware
 */
#include "gmxpre.h"

#include "gromacs/hardware/cpuinfo.h"

#include "config.h"

#include <gtest/gtest.h>

namespace
{

TEST(CpuInfoTest, SupportLevel)
{
    // There is no way we can compare to any reference data since that
    // depends on the architecture, but we can at least make sure that it
    // works to execute the tests

    gmx::CpuInfo c(gmx::CpuInfo::detect());

    std::string commonMsg =
            "\nGROMACS might still work, but it will likely hurt your performance."
            "\nPlease make a post at the GROMACS development forum at"
            "\nhttps://gromacs.bioexcel.eu/c/gromacs-developers/10 so we can try to fix it.";

    // It is not the end of the world if any of these tests fail (Gromacs will
    // work fine without cpuinfo), but we might as well flag it so we add it to
    // our detection code
    EXPECT_GT(c.supportLevel(), gmx::CpuInfo::SupportLevel::None)
            << "No CPU information at all could be detected. " << commonMsg << std::endl;

#if GMX_TARGET_X86
    EXPECT_GE(c.supportLevel(), gmx::CpuInfo::SupportLevel::Features)
            << "No CPU features could be detected. " << commonMsg << std::endl;
#endif

    if (c.supportLevel() >= gmx::CpuInfo::SupportLevel::LogicalProcessorInfo)
    {
        // Make sure assigned numbers are reasonable if we have them
        for (const auto& l : c.logicalProcessors())
        {
            EXPECT_GE(l.packageIdInMachine, 0)
                    << "Impossible package index for logical processor. " << commonMsg << std::endl;
            EXPECT_GE(l.coreIdInPackage, 0)
                    << "Impossible core index for logical processor. " << commonMsg << std::endl;
            EXPECT_GE(l.puIdInCore, 0)
                    << "Impossible pu index for logical processor. " << commonMsg << std::endl;
        }
    }
}

} // namespace
