/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2022- The GROMACS Authors
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
 * Implements test of index generation routines
 *
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 * \ingroup module_topology
 */
#include "gmxpre.h"

#include "gromacs/topology/index.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "gromacs/trajectoryanalysis/topologyinformation.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/smalloc.h"

#include "testutils/cmdlinetest.h"
#include "testutils/filematchers.h"
#include "testutils/refdata.h"
#include "testutils/testfilemanager.h"
#include "testutils/textblockmatchers.h"

namespace gmx
{
namespace test
{
namespace
{

void checkIndexGroup(TestReferenceChecker* checker, const IndexGroup& group)
{
    TestReferenceChecker compound(checker->checkCompound("Group", nullptr));
    compound.checkString(group.name, "GroupName");
    compound.checkSequence(group.particleIndices.begin(), group.particleIndices.end(), "Entries");
}

void checkBlocks(TestReferenceChecker* checker, ArrayRef<const IndexGroup> blocks)
{
    TestReferenceChecker compound(checker->checkCompound("Blocks", nullptr));
    compound.checkInteger(blocks.size(), "Number");
    compound.checkSequence(blocks.begin(), blocks.end(), "Index", checkIndexGroup);
}

void compareBlocks(ArrayRef<const IndexGroup> one, ArrayRef<const IndexGroup> two)
{
    ASSERT_EQ(one.size(), two.size());
    for (int i = 0; i < gmx::ssize(one); ++i)
    {
        EXPECT_EQ(one[i].name, two[i].name);
        ASSERT_EQ(one[i].particleIndices.size(), two[i].particleIndices.size());
        for (int j = 0; j < gmx::ssize(one[i].particleIndices); ++j)
        {
            EXPECT_EQ(one[i].particleIndices[j], two[i].particleIndices[j]);
        }
    }
}

void checkFileMatch(TestReferenceChecker* checker, const std::string& fileName, const std::string& fullPath)
{
    TestReferenceChecker fileChecker(checker->checkCompound("File", fileName.c_str()));
    auto                 matcher = TextFileMatch(ExactTextMatch()).createFileMatcher();
    matcher->checkFile(fullPath, &fileChecker);
}

class IndexTest : public ::testing::Test
{
public:
    IndexTest();

    //! Handle to atoms from topology.
    const t_atoms* atoms() { return topInfo_.atoms(); }
    //! Handle to checker.
    TestReferenceChecker* checker() { return &checker_; }
    //! Handle to file manager.
    TestFileManager* manager() { return &manager_; }

private:
    //! Input structure data.
    TopologyInformation topInfo_;
    //! File manager for test.
    TestFileManager manager_;
    //! Handler for reference data.
    TestReferenceData data_;
    //! Handler for checking test data.
    TestReferenceChecker checker_;
};

IndexTest::IndexTest() : checker_(data_.rootChecker())
{
    // When we have many test cases using this class, refactor to fill
    // a static topInfo only once, in SetUpTestSuite()
    topInfo_.fillFromInputFile(manager()->getInputFilePath("lysozyme.gro").u8string());
}

TEST_F(IndexTest, AnalyseWorksDefaultGroups)
{
    auto indexGroups = analyse(atoms(), false, false);
    checkBlocks(checker(), indexGroups);
}

TEST_F(IndexTest, WriteIndexWorks)
{
    auto        indexGroups = analyse(atoms(), false, false);
    std::string fileName    = "out.ndx";
    std::string fullPath    = manager()->getTemporaryFilePath(fileName).u8string();
    write_index(fullPath.c_str(), indexGroups, false, atoms()->nr);
    checkFileMatch(checker(), fileName, fullPath);
}

TEST_F(IndexTest, WriteAndReadIndexWorks)
{
    auto        indexGroups = analyse(atoms(), false, false);
    std::string fileName    = "out.ndx";
    std::string fullPath    = manager()->getTemporaryFilePath(fileName).u8string();
    write_index(fullPath.c_str(), indexGroups, false, atoms()->nr);
    auto newIndex = init_index(fullPath.c_str());
    compareBlocks(indexGroups, newIndex);
}

} // namespace

} // namespace test

} // namespace gmx
