/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2019- The GROMACS Authors
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

/*! \libinternal \file
 * \brief
 * Helper for generating reusuable TPR files for tests within the same test binary.
 *
 * \ingroup module_testutils
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 */
#ifndef GMX_TESTUTILS_TPRFILEGENERATOR_H
#define GMX_TESTUTILS_TPRFILEGENERATOR_H

#include <memory>
#include <string>

#include "testutils/testfilemanager.h"

namespace gmx
{
namespace test
{

class TestFileManager;

/*! \libinternal \brief
 * Helper to bundle generated TPR and the file manager to clean it up.
 */
class TprAndFileManager
{
public:
    /*! \brief
     * Generates the file when needed.
     *
     * \param[in] name The basename of the input files and the generated TPR.
     */
    TprAndFileManager(const std::string& name);
    //! Access to the string.
    const std::string& tprName() const { return tprFileName_; }

private:
    //! Tpr file name.
    std::string tprFileName_;
    //! Filemanager, needed to clean up the file later.
    TestFileManager fileManager_;
};

} // namespace test
} // namespace gmx

#endif
