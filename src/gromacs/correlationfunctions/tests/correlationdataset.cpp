/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2014- The GROMACS Authors
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
 * Implements helper class for autocorrelation tests
 *
 * \author Anders G&auml;rden&auml;s <anders.gardenas@gmail.com>
 * \ingroup module_correlationfunctions
 */
#include "gmxpre.h"

#include "correlationdataset.h"

#include <cmath>

#include <sstream>

#include "gromacs/fileio/xvgr.h"
#include "gromacs/utility/smalloc.h"

#include "testutils/testfilemanager.h"

CorrelationDataSet::CorrelationDataSet(const std::string& fileName)
{
    std::string fileNm = gmx::test::TestFileManager::getInputFilePath(fileName).u8string();
    nrLines_           = read_xvg(fileNm.c_str(), &tempValues_, &nrColumns_);

    dt_        = tempValues_[0][1] - tempValues_[0][0];
    startTime_ = tempValues_[0][0];
    endTime_   = tempValues_[0][nrLines_ - 1];
}

CorrelationDataSet::~CorrelationDataSet()
{
    // Allocated in read_xvg, destroyed here.
    for (int i = 0; i < nrColumns_; i++)
    {
        sfree(tempValues_[i]);
        tempValues_[i] = nullptr;
    }
    sfree(tempValues_);
    tempValues_ = nullptr;
}

real CorrelationDataSet::getValue(int set, int time) const
{
    if (set + 1 < nrColumns_)
    {
        return tempValues_[set + 1][time];
    }
    else
    {
        return 0;
    }
}
