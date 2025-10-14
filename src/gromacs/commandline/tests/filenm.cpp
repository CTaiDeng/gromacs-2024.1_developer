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

/*! \internal \file
 * \brief
 * Tests filename-handling functionality.
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \ingroup module_commandline
 */
#include "gmxpre.h"

#include "gromacs/commandline/filenm.h"

#include <string>

#include <gtest/gtest.h>

namespace gmx
{
namespace test
{
namespace
{

TEST(OutputNamesTest, CanBeSuffixed)
{
    std::vector<t_filenm> filenames = { { efTRR, nullptr, nullptr, ffREAD, { "input.trr" } },
                                        { efTRR, nullptr, nullptr, ffWRITE, { "output.trr" } },
                                        { efCPT, nullptr, nullptr, ffWRITE, { "output.cpt" } } };
    add_suffix_to_output_names(filenames.data(), filenames.size(), "_suffix");
    EXPECT_EQ(filenames[0].filenames[0], "input.trr");
    EXPECT_EQ(filenames[1].filenames[0], "output_suffix.trr");
    EXPECT_EQ(filenames[2].filenames[0], "output.cpt");
}

TEST(OutputNamesTest, HasSuffixFromNoAppend)
{
    EXPECT_FALSE(hasSuffixFromNoAppend("output"));
    EXPECT_FALSE(hasSuffixFromNoAppend("output.log"));
    EXPECT_TRUE(hasSuffixFromNoAppend("output.part0002.log"));
    EXPECT_TRUE(hasSuffixFromNoAppend("output.equil.part0002.log"));
    EXPECT_TRUE(hasSuffixFromNoAppend("output.equil.part0001.part0002.log"));
    EXPECT_FALSE(hasSuffixFromNoAppend("output.part0002"));
    EXPECT_FALSE(hasSuffixFromNoAppend("part0002.log"));
    EXPECT_FALSE(hasSuffixFromNoAppend("output.part02.log"));
    EXPECT_FALSE(hasSuffixFromNoAppend("output.part002.log"));
}

TEST(OutputNamesTest, CanHavePartNumberAdded)
{
    std::vector<t_filenm> filenames = {
        { efLOG, nullptr, nullptr, ffWRITE, { "output.log" } },
        { efLOG, nullptr, nullptr, ffWRITE, { "output.part0002.log" } },
        { efLOG, nullptr, nullptr, ffWRITE, { "output.equil.part0002.log" } },
        { efLOG, nullptr, nullptr, ffWRITE, { "output.part0001.part0002.log" } },
        { efLOG, nullptr, nullptr, ffWRITE, { "output.equil.part0001.part0002.log" } },
        { efLOG, nullptr, nullptr, ffWRITE, { "output.part0002" } },
        { efLOG, nullptr, nullptr, ffWRITE, { "part0002.log" } },
        { efLOG, nullptr, nullptr, ffWRITE, { "output.part02.log" } },
        { efLOG, nullptr, nullptr, ffWRITE, { "output.part002.log" } },
        { efLOG, nullptr, nullptr, ffWRITE, { "output" } }
    };
    add_suffix_to_output_names(filenames.data(), filenames.size(), ".part0003");
    EXPECT_EQ(filenames[0].filenames[0], "output.part0003.log");
    EXPECT_EQ(filenames[1].filenames[0], "output.part0003.log");
    EXPECT_EQ(filenames[2].filenames[0], "output.equil.part0003.log");
    EXPECT_EQ(filenames[3].filenames[0], "output.part0003.log");
    EXPECT_EQ(filenames[4].filenames[0], "output.equil.part0003.log");
    EXPECT_EQ(filenames[5].filenames[0], "output.part0003.part0002");
    EXPECT_EQ(filenames[6].filenames[0], "part0002.part0003.log");
    EXPECT_EQ(filenames[7].filenames[0], "output.part02.part0003.log");
    EXPECT_EQ(filenames[8].filenames[0], "output.part002.part0003.log");
    EXPECT_EQ(filenames[9].filenames[0], "output.part0003");
}

} // namespace
} // namespace test
} // namespace gmx
