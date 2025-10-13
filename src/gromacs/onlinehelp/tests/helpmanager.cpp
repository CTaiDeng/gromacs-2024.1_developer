/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2012- The GROMACS Authors
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
 * Tests for help topic management and help topic formatting.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_onlinehelp
 */
#include "gmxpre.h"

#include "gromacs/onlinehelp/helpmanager.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "gromacs/onlinehelp/helptopic.h"
#include "gromacs/onlinehelp/helpwritercontext.h"
#include "gromacs/onlinehelp/tests/mock_helptopic.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/stringstream.h"
#include "gromacs/utility/textwriter.h"

#include "testutils/stringtest.h"
#include "testutils/testasserts.h"

namespace
{

using gmx::test::MockHelpTopic;

class HelpTestBase : public gmx::test::StringTestBase
{
public:
    HelpTestBase();

    MockHelpTopic           rootTopic_;
    gmx::StringOutputStream helpFile_;
    gmx::TextWriter         writer_;
    gmx::HelpWriterContext  context_;
    gmx::HelpManager        manager_;
};

HelpTestBase::HelpTestBase() :
    rootTopic_("", nullptr, "Root topic text"),
    writer_(&helpFile_),
    context_(&writer_, gmx::eHelpOutputFormat_Console),
    manager_(rootTopic_, context_)
{
}

/********************************************************************
 * Tests for HelpManager
 */

//! Test fixture for gmx::HelpManager.
typedef HelpTestBase HelpManagerTest;

TEST_F(HelpManagerTest, HandlesRootTopic)
{
    using ::testing::_;
    EXPECT_CALL(rootTopic_, writeHelp(_));
    manager_.writeCurrentTopic();
}

TEST_F(HelpManagerTest, HandlesSubTopics)
{
    MockHelpTopic& first    = rootTopic_.addSubTopic("first", "First topic", nullptr);
    MockHelpTopic& firstSub = first.addSubTopic("firstsub", "First subtopic", nullptr);
    rootTopic_.addSubTopic("second", "Second topic", nullptr);

    using ::testing::_;
    EXPECT_CALL(firstSub, writeHelp(_));
    ASSERT_NO_THROW_GMX(manager_.enterTopic("first"));
    ASSERT_NO_THROW_GMX(manager_.enterTopic("firstsub"));
    manager_.writeCurrentTopic();
}

TEST_F(HelpManagerTest, HandlesInvalidTopics)
{
    MockHelpTopic& first = rootTopic_.addSubTopic("first", "First topic", nullptr);
    first.addSubTopic("firstsub", "First subtopic", nullptr);
    rootTopic_.addSubTopic("second", "Second topic", nullptr);

    ASSERT_THROW_GMX(manager_.enterTopic("unknown"), gmx::InvalidInputError);
    ASSERT_NO_THROW_GMX(manager_.enterTopic("first"));
    ASSERT_THROW_GMX(manager_.enterTopic("unknown"), gmx::InvalidInputError);
    ASSERT_THROW_GMX(manager_.enterTopic("second"), gmx::InvalidInputError);
    ASSERT_NO_THROW_GMX(manager_.enterTopic("firstsub"));
}

/********************************************************************
 * Tests for help topic formatting
 */

struct TestHelpText
{
    static const char        name[];
    static const char        title[];
    static const char* const text[];
};

const char        TestHelpText::name[]  = "testtopic";
const char        TestHelpText::title[] = "Topic title";
const char* const TestHelpText::text[]  = { "Test topic text.[PAR]", "Another paragraph of text." };

class HelpTopicFormattingTest : public HelpTestBase
{
public:
    void checkHelpFormatting();
};

void HelpTopicFormattingTest::checkHelpFormatting()
{
    ASSERT_NO_THROW_GMX(manager_.enterTopic("testtopic"));
    ASSERT_NO_THROW_GMX(manager_.writeCurrentTopic());
    helpFile_.close();

    checkText(helpFile_.toString(), "HelpText");
}

TEST_F(HelpTopicFormattingTest, FormatsSimpleTopic)
{
    rootTopic_.addSubTopic(gmx::HelpTopicPointer(new gmx::SimpleHelpTopic<TestHelpText>));
    checkHelpFormatting();
}

TEST_F(HelpTopicFormattingTest, FormatsCompositeTopicWithSubTopics)
{
    gmx::CompositeHelpTopicPointer topic(new gmx::CompositeHelpTopic<TestHelpText>);
    MockHelpTopic::addSubTopic(topic.get(), "subtopic", "First subtopic", "Text");
    MockHelpTopic::addSubTopic(topic.get(), "other", "Second subtopic", "Text");
    rootTopic_.addSubTopic(std::move(topic));
    checkHelpFormatting();
}

} // namespace
