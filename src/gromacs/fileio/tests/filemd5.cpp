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

/*! \internal \file
 * \brief
 * Tests for routines for computing MD5 sums on files.
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \ingroup module_fileio
 */
#include "gmxpre.h"

#include <cstdio>

#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "gromacs/fileio/gmxfio.h"
#include "gromacs/utility/arrayref.h"

#include "testutils/refdata.h"
#include "testutils/testfilemanager.h"

namespace gmx
{
namespace test
{
namespace
{

class FileMD5Test : public ::testing::Test
{
public:
    void prepareFile(int lengthInBytes) const
    {
        // Fill some memory with some arbitrary bits.
        std::vector<char> data(lengthInBytes);
        std::iota(data.begin(), data.end(), 1);
        // Binary mode ensures it works the same on all OS
        FILE* fp = fopen(filename_.c_str(), "wb");
        fwrite(data.data(), sizeof(char), data.size(), fp);
        fclose(fp);
    }
    ~FileMD5Test() override
    {
        if (file_)
        {
            gmx_fio_close(file_);
        }
    }
    TestFileManager fileManager_;
    // Make sure the file extension is one that gmx_fio_open will
    // recognize to open as binary.
    std::string filename_ = fileManager_.getTemporaryFilePath("data.edr").u8string();
    t_fileio*   file_     = nullptr;
};

TEST_F(FileMD5Test, CanComputeMD5)
{
    prepareFile(1000);
    file_ = gmx_fio_open(filename_.c_str(), "r+");

    std::array<unsigned char, 16> digest = { 0 };
    // Chosen to be less than the full file length
    gmx_off_t offset             = 64;
    gmx_off_t expectedLength     = 64;
    gmx_off_t lengthActuallyRead = gmx_fio_get_file_md5(file_, offset, &digest);

    EXPECT_EQ(expectedLength, lengthActuallyRead);
    // Did we compute an actual reproducible checksum?
    auto total = std::accumulate(digest.begin(), digest.end(), 0);
    EXPECT_EQ(2111, total);
}

TEST_F(FileMD5Test, ReturnsErrorIfFileModeIsWrong)
{
    prepareFile(1000);
    file_ = gmx_fio_open(filename_.c_str(), "r");

    std::array<unsigned char, 16> digest;
    gmx_off_t                     offset             = 100;
    gmx_off_t                     lengthActuallyRead = gmx_fio_get_file_md5(file_, offset, &digest);
    EXPECT_EQ(-1, lengthActuallyRead);
}

} // namespace
} // namespace test
} // namespace gmx
