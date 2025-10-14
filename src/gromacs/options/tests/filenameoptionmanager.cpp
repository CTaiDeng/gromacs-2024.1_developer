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
 * Tests file name option implementation dependent on gmx::FileNameOptionManager.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_options
 */
#include "gmxpre.h"

#include "gromacs/options/filenameoptionmanager.h"

#include <gtest/gtest.h>

#include "gromacs/fileio/filetypes.h"
#include "gromacs/options/filenameoption.h"
#include "gromacs/options/options.h"
#include "gromacs/options/optionsassigner.h"
#include "gromacs/utility/exceptions.h"

#include "testutils/testasserts.h"
#include "testutils/testfileredirector.h"

namespace
{

using gmx::FileNameOption;

class FileNameOptionManagerTest : public ::testing::Test
{
public:
    FileNameOptionManagerTest()
    {
        manager_.setInputRedirector(&redirector_);
        options_.addManager(&manager_);
    }

    void addExistingFile(const char* filename) { redirector_.addExistingFile(filename); }

    gmx::test::TestFileInputRedirector redirector_;
    gmx::FileNameOptionManager         manager_;
    gmx::Options                       options_;
};

/********************************************************************
 * Actual tests
 */

TEST_F(FileNameOptionManagerTest, AddsMissingExtension)
{
    std::string value;
    ASSERT_NO_THROW_GMX(options_.addOption(
            FileNameOption("f").store(&value).filetype(gmx::OptionFileType::Trajectory).outputFile()));

    gmx::OptionsAssigner assigner(&options_);
    EXPECT_NO_THROW_GMX(assigner.start());
    EXPECT_NO_THROW_GMX(assigner.startOption("f"));
    EXPECT_NO_THROW_GMX(assigner.appendValue("testfile"));
    EXPECT_NO_THROW_GMX(assigner.finishOption());
    EXPECT_NO_THROW_GMX(assigner.finish());
    EXPECT_NO_THROW_GMX(options_.finish());

    EXPECT_EQ("testfile.xtc", value);
}

TEST_F(FileNameOptionManagerTest, AddsMissingCustomDefaultExtension)
{
    std::string value;
    ASSERT_NO_THROW_GMX(options_.addOption(
            FileNameOption("f").store(&value).filetype(gmx::OptionFileType::Trajectory).outputFile().defaultType(efPDB)));

    gmx::OptionsAssigner assigner(&options_);
    EXPECT_NO_THROW_GMX(assigner.start());
    EXPECT_NO_THROW_GMX(assigner.startOption("f"));
    EXPECT_NO_THROW_GMX(assigner.appendValue("testfile"));
    EXPECT_NO_THROW_GMX(assigner.finishOption());
    EXPECT_NO_THROW_GMX(assigner.finish());
    EXPECT_NO_THROW_GMX(options_.finish());

    EXPECT_EQ("testfile.pdb", value);
}

TEST_F(FileNameOptionManagerTest, GivesErrorOnMissingInputFile)
{
    std::string value;
    ASSERT_NO_THROW_GMX(options_.addOption(
            FileNameOption("f").store(&value).filetype(gmx::OptionFileType::AtomIndex).inputFile()));
    EXPECT_TRUE(value.empty());

    gmx::OptionsAssigner assigner(&options_);
    EXPECT_NO_THROW_GMX(assigner.start());
    EXPECT_NO_THROW_GMX(assigner.startOption("f"));
    EXPECT_THROW_GMX(assigner.appendValue("missing.ndx"), gmx::InvalidInputError);
    EXPECT_NO_THROW_GMX(assigner.finishOption());
    EXPECT_NO_THROW_GMX(assigner.finish());
    EXPECT_NO_THROW_GMX(options_.finish());

    EXPECT_TRUE(value.empty());
}

TEST_F(FileNameOptionManagerTest, GivesErrorOnMissingGenericInputFile)
{
    std::string value;
    ASSERT_NO_THROW_GMX(options_.addOption(
            FileNameOption("f").store(&value).filetype(gmx::OptionFileType::Trajectory).inputFile()));
    EXPECT_TRUE(value.empty());

    gmx::OptionsAssigner assigner(&options_);
    EXPECT_NO_THROW_GMX(assigner.start());
    EXPECT_NO_THROW_GMX(assigner.startOption("f"));
    EXPECT_THROW_GMX(assigner.appendValue("missing.trr"), gmx::InvalidInputError);
    EXPECT_NO_THROW_GMX(assigner.finishOption());
    EXPECT_NO_THROW_GMX(assigner.finish());
    EXPECT_NO_THROW_GMX(options_.finish());

    EXPECT_TRUE(value.empty());
}

TEST_F(FileNameOptionManagerTest, GivesErrorOnMissingDefaultInputFile)
{
    std::string value;
    ASSERT_NO_THROW_GMX(options_.addOption(
            FileNameOption("f").store(&value).filetype(gmx::OptionFileType::AtomIndex).inputFile().defaultBasename("missing")));

    gmx::OptionsAssigner assigner(&options_);
    EXPECT_NO_THROW_GMX(assigner.start());
    EXPECT_NO_THROW_GMX(assigner.startOption("f"));
    EXPECT_NO_THROW_GMX(assigner.finishOption());
    EXPECT_NO_THROW_GMX(assigner.finish());
    EXPECT_THROW_GMX(options_.finish(), gmx::InvalidInputError);
}

TEST_F(FileNameOptionManagerTest, GivesErrorOnMissingRequiredInputFile)
{
    std::string value;
    ASSERT_NO_THROW_GMX(options_.addOption(FileNameOption("f")
                                                   .store(&value)
                                                   .required()
                                                   .filetype(gmx::OptionFileType::AtomIndex)
                                                   .inputFile()
                                                   .defaultBasename("missing")));
    EXPECT_EQ("missing.ndx", value);

    gmx::OptionsAssigner assigner(&options_);
    EXPECT_NO_THROW_GMX(assigner.start());
    EXPECT_NO_THROW_GMX(assigner.finish());
    EXPECT_THROW_GMX(options_.finish(), gmx::InvalidInputError);
}

TEST_F(FileNameOptionManagerTest, AcceptsMissingInputFileIfSpecified)
{
    std::string value;
    ASSERT_NO_THROW_GMX(options_.addOption(
            FileNameOption("f").store(&value).filetype(gmx::OptionFileType::AtomIndex).inputFile().allowMissing()));
    EXPECT_TRUE(value.empty());

    gmx::OptionsAssigner assigner(&options_);
    EXPECT_NO_THROW_GMX(assigner.start());
    EXPECT_NO_THROW_GMX(assigner.startOption("f"));
    EXPECT_NO_THROW_GMX(assigner.appendValue("missing.ndx"));
    EXPECT_NO_THROW_GMX(assigner.finishOption());
    EXPECT_NO_THROW_GMX(assigner.finish());
    EXPECT_NO_THROW_GMX(options_.finish());

    EXPECT_EQ("missing.ndx", value);
}

TEST_F(FileNameOptionManagerTest, AcceptsMissingDefaultInputFileIfSpecified)
{
    std::string value;
    ASSERT_NO_THROW_GMX(options_.addOption(FileNameOption("f")
                                                   .store(&value)
                                                   .filetype(gmx::OptionFileType::AtomIndex)
                                                   .inputFile()
                                                   .defaultBasename("missing")
                                                   .allowMissing()));

    gmx::OptionsAssigner assigner(&options_);
    EXPECT_NO_THROW_GMX(assigner.start());
    EXPECT_NO_THROW_GMX(assigner.startOption("f"));
    EXPECT_NO_THROW_GMX(assigner.finishOption());
    EXPECT_NO_THROW_GMX(assigner.finish());
    EXPECT_NO_THROW_GMX(options_.finish());

    EXPECT_EQ("missing.ndx", value);
}

TEST_F(FileNameOptionManagerTest, AcceptsMissingRequiredInputFileIfSpecified)
{
    std::string value;
    ASSERT_NO_THROW_GMX(options_.addOption(FileNameOption("f")
                                                   .store(&value)
                                                   .required()
                                                   .filetype(gmx::OptionFileType::AtomIndex)
                                                   .inputFile()
                                                   .defaultBasename("missing")
                                                   .allowMissing()));
    EXPECT_EQ("missing.ndx", value);

    gmx::OptionsAssigner assigner(&options_);
    EXPECT_NO_THROW_GMX(assigner.start());
    EXPECT_NO_THROW_GMX(assigner.finish());
    EXPECT_NO_THROW_GMX(options_.finish());

    EXPECT_EQ("missing.ndx", value);
}

TEST_F(FileNameOptionManagerTest, AddsMissingExtensionBasedOnExistingFile)
{
    addExistingFile("testfile.trr");

    std::string value;
    ASSERT_NO_THROW_GMX(options_.addOption(
            FileNameOption("f").store(&value).filetype(gmx::OptionFileType::Trajectory).inputFile()));

    gmx::OptionsAssigner assigner(&options_);
    EXPECT_NO_THROW_GMX(assigner.start());
    EXPECT_NO_THROW_GMX(assigner.startOption("f"));
    EXPECT_NO_THROW_GMX(assigner.appendValue("testfile"));
    EXPECT_NO_THROW_GMX(assigner.finishOption());
    EXPECT_NO_THROW_GMX(assigner.finish());
    EXPECT_NO_THROW_GMX(options_.finish());

    EXPECT_EQ("testfile.trr", value);
}

TEST_F(FileNameOptionManagerTest, AddsMissingExtensionForRequiredDefaultNameBasedOnExistingFile)
{
    addExistingFile("testfile.trr");

    std::string value;
    ASSERT_NO_THROW_GMX(options_.addOption(FileNameOption("f")
                                                   .store(&value)
                                                   .required()
                                                   .filetype(gmx::OptionFileType::Trajectory)
                                                   .inputFile()
                                                   .defaultBasename("testfile")));
    EXPECT_EQ("testfile.xtc", value);

    gmx::OptionsAssigner assigner(&options_);
    EXPECT_NO_THROW_GMX(assigner.start());
    EXPECT_NO_THROW_GMX(assigner.startOption("f"));
    EXPECT_NO_THROW_GMX(assigner.finishOption());
    EXPECT_NO_THROW_GMX(assigner.finish());
    EXPECT_NO_THROW_GMX(options_.finish());

    EXPECT_EQ("testfile.trr", value);
}

TEST_F(FileNameOptionManagerTest, AddsMissingExtensionForOptionalDefaultNameBasedOnExistingFile)
{
    addExistingFile("testfile.trr");

    std::string value;
    ASSERT_NO_THROW_GMX(options_.addOption(
            FileNameOption("f").store(&value).filetype(gmx::OptionFileType::Trajectory).inputFile().defaultBasename("testfile")));

    gmx::OptionsAssigner assigner(&options_);
    EXPECT_NO_THROW_GMX(assigner.start());
    EXPECT_NO_THROW_GMX(assigner.startOption("f"));
    EXPECT_NO_THROW_GMX(assigner.finishOption());
    EXPECT_NO_THROW_GMX(assigner.finish());
    EXPECT_NO_THROW_GMX(options_.finish());

    EXPECT_EQ("testfile.trr", value);
}

TEST_F(FileNameOptionManagerTest, AddsMissingExtensionForRequiredFromDefaultNameOptionBasedOnExistingFile)
{
    addExistingFile("testfile.trr");

    std::string value;
    ASSERT_NO_THROW_GMX(options_.addOption(FileNameOption("f")
                                                   .store(&value)
                                                   .required()
                                                   .filetype(gmx::OptionFileType::Trajectory)
                                                   .inputFile()
                                                   .defaultBasename("foo")));
    ASSERT_NO_THROW_GMX(manager_.addDefaultFileNameOption(&options_, "deffnm"));
    EXPECT_EQ("foo.xtc", value);

    gmx::OptionsAssigner assigner(&options_);
    EXPECT_NO_THROW_GMX(assigner.start());
    EXPECT_NO_THROW_GMX(assigner.startOption("deffnm"));
    EXPECT_NO_THROW_GMX(assigner.appendValue("testfile"));
    EXPECT_NO_THROW_GMX(assigner.finishOption());
    EXPECT_NO_THROW_GMX(assigner.finish());
    EXPECT_NO_THROW_GMX(options_.finish());

    EXPECT_EQ("testfile.trr", value);
}

TEST_F(FileNameOptionManagerTest, AddsMissingExtensionForOptionalFromDefaultNameOptionBasedOnExistingFile)
{
    addExistingFile("testfile.trr");

    std::string value;
    ASSERT_NO_THROW_GMX(options_.addOption(
            FileNameOption("f").store(&value).filetype(gmx::OptionFileType::Trajectory).inputFile().defaultBasename("foo")));
    ASSERT_NO_THROW_GMX(manager_.addDefaultFileNameOption(&options_, "deffnm"));

    gmx::OptionsAssigner assigner(&options_);
    EXPECT_NO_THROW_GMX(assigner.start());
    EXPECT_NO_THROW_GMX(assigner.startOption("deffnm"));
    EXPECT_NO_THROW_GMX(assigner.appendValue("testfile"));
    EXPECT_NO_THROW_GMX(assigner.finishOption());
    EXPECT_NO_THROW_GMX(assigner.startOption("f"));
    EXPECT_NO_THROW_GMX(assigner.finishOption());
    EXPECT_NO_THROW_GMX(assigner.finish());
    EXPECT_NO_THROW_GMX(options_.finish());

    EXPECT_EQ("testfile.trr", value);
}

TEST_F(FileNameOptionManagerTest, DefaultNameOptionWorksWithoutInputChecking)
{
    std::string value;
    ASSERT_NO_THROW_GMX(manager_.disableInputOptionChecking(true));
    ASSERT_NO_THROW_GMX(options_.addOption(FileNameOption("f")
                                                   .store(&value)
                                                   .required()
                                                   .filetype(gmx::OptionFileType::AtomIndex)
                                                   .inputFile()
                                                   .defaultBasename("default")
                                                   .allowMissing()));
    ASSERT_NO_THROW_GMX(manager_.addDefaultFileNameOption(&options_, "deffnm"));
    EXPECT_EQ("default.ndx", value);

    gmx::OptionsAssigner assigner(&options_);
    EXPECT_NO_THROW_GMX(assigner.start());
    EXPECT_NO_THROW_GMX(assigner.startOption("deffnm"));
    EXPECT_NO_THROW_GMX(assigner.appendValue("missing"));
    EXPECT_NO_THROW_GMX(assigner.finishOption());
    EXPECT_NO_THROW_GMX(assigner.finish());
    EXPECT_NO_THROW_GMX(options_.finish());

    EXPECT_EQ("missing.ndx", value);
}

} // namespace
