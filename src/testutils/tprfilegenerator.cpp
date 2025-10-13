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

/*! \internal \file
 * \brief
 * Implements helper for generating reusuable TPR files for tests within the same test binary.
 *
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 * \ingroup module_testutils
 */

#include "gmxpre.h"

#include "testutils/tprfilegenerator.h"

#include "gromacs/gmxpreprocess/grompp.h"
#include "gromacs/utility/textwriter.h"

#include "testutils/cmdlinetest.h"

namespace gmx
{
namespace test
{

TprAndFileManager::TprAndFileManager(const std::string& name)
{
    const std::string mdpInputFileName = fileManager_.getTemporaryFilePath(name + ".mdp").u8string();
    gmx::TextWriter::writeFileFromString(mdpInputFileName, "");
    tprFileName_ = fileManager_.getTemporaryFilePath(name + ".tpr").u8string();
    {
        CommandLine caller;
        caller.append("grompp");
        caller.addOption("-f", mdpInputFileName);
        caller.addOption("-p", TestFileManager::getInputFilePath(name + ".top").u8string());
        caller.addOption("-c", TestFileManager::getInputFilePath(name + ".pdb").u8string());
        caller.addOption("-o", tprFileName_);
        EXPECT_EQ(0, gmx_grompp(caller.argc(), caller.argv()));
    }
}

} // namespace test
} // namespace gmx
