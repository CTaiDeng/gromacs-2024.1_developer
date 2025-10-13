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
 * Implements classes in mock_helptopic.h.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_onlinehelp
 */
#include "gmxpre.h"

#include "mock_helptopic.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "gromacs/utility/stringutil.h"

namespace gmx
{
namespace test
{

/********************************************************************
 * MockHelpTopic
 */

// static
MockHelpTopic& MockHelpTopic::addSubTopic(gmx::AbstractCompositeHelpTopic* parent,
                                          const char*                      name,
                                          const char*                      title,
                                          const char*                      text)
{
    MockHelpTopic* topic = new MockHelpTopic(name, title, text);
    parent->addSubTopic(gmx::HelpTopicPointer(topic));
    return *topic;
}

MockHelpTopic::MockHelpTopic(const char* name, const char* title, const char* text) :
    name_(name), title_(title), text_(text != nullptr ? text : "")
{
    if (!isNullOrEmpty(text))
    {
        using ::testing::_;
        using ::testing::Invoke;
        ON_CALL(*this, writeHelp(_)).WillByDefault(Invoke(this, &MockHelpTopic::writeHelpBase));
    }
}

MockHelpTopic::~MockHelpTopic() {}

const char* MockHelpTopic::name() const
{
    return name_;
}

const char* MockHelpTopic::title() const
{
    return title_;
}

std::string MockHelpTopic::helpText() const
{
    return text_;
}

MockHelpTopic& MockHelpTopic::addSubTopic(const char* name, const char* title, const char* text)
{
    return addSubTopic(this, name, title, text);
}

} // namespace test
} // namespace gmx
