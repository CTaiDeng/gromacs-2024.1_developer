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
 * Tests for outputmanager
 *
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 * \ingroup module_coordinateio
 */


#include "gmxpre.h"

#include "config.h"

#include <utility>

#include "gromacs/coordinateio/coordinatefile.h"
#include "gromacs/coordinateio/tests/coordinate_test.h"

namespace gmx
{

namespace test
{

/*!\brief
 * Test fixture to test different file types are supported by the CoordinateFile.
 */
class TrajectoryFrameWriterTest : public ModuleTest
{
public:
    /*! \brief
     * Test basic behaviour without special requirements.
     *
     * \param[in] filename Name for output file.
     */
    void basicTest(const char* filename)
    {
        addTopology();

        OutputRequirements requirements;

        runTest(filename, requirements);
    }
    /*! \brief
     * Test with extra requirements.
     *
     * \param[in] filename Name for output file.
     * \param[in] requirements Specify extra reqs for output.
     */
    void testWithRequirements(const char* filename, const OutputRequirements& requirements)
    {
        addTopology();
        runTest(filename, requirements);
    }
};

TEST_P(TrajectoryFrameWriterTest, WorksWithFormats)
{
    EXPECT_NO_THROW(basicTest(GetParam()));
}

TEST_F(TrajectoryFrameWriterTest, RejectsWrongFiletype)
{
    EXPECT_THROW(basicTest("test.xvg"), InvalidInputError);
}

TEST_F(TrajectoryFrameWriterTest, BuilderFailsWithPdbAndNoAtoms)
{
    OutputRequirements requirements;
    requirements.atoms = ChangeAtomsType::Never;
    EXPECT_THROW(testWithRequirements("test.pdb", requirements), InconsistentInputError);
}

TEST_F(TrajectoryFrameWriterTest, BuilderFailsWithGroAndNoAtoms)
{
    OutputRequirements requirements;
    requirements.atoms = ChangeAtomsType::Never;
    EXPECT_THROW(testWithRequirements("test.gro", requirements), InconsistentInputError);
}

TEST_F(TrajectoryFrameWriterTest, BuilderImplictlyAddsAtoms)
{
    OutputRequirements requirements;
    requirements.atoms = ChangeAtomsType::PreservedIfPresent;
    {
        EXPECT_NO_THROW(testWithRequirements("test.pdb", requirements));
    }
    {
        EXPECT_NO_THROW(testWithRequirements("test.gro", requirements));
    }
}

#if GMX_USE_TNG
TEST_F(TrajectoryFrameWriterTest, TNGOutputWorks)
{
    OutputRequirements requirements;
    runTest("test.tng", requirements);
}
#endif

/*!\brief
 * Character array of different file names to test.
 */
const char* const trajectoryFileNames[] = { "spc2-traj.trr",
#if GMX_USE_TNG
                                            "spc2-traj.tng",
#endif
                                            "spc2-traj.xtc", "spc2-traj.pdb",
                                            "spc2-traj.gro", "spc2-traj.g96" };

INSTANTIATE_TEST_SUITE_P(CoordinateFileFileFormats,
                         TrajectoryFrameWriterTest,
                         ::testing::ValuesIn(trajectoryFileNames));

} // namespace test

} // namespace gmx
