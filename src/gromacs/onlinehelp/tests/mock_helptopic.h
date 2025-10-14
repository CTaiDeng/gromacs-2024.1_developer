/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2012- The GROMACS Authors
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
 * Declares mock implementation of gmx::IHelpTopic.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \inlibraryapi
 * \ingroup module_onlinehelp
 */
#ifndef GMX_ONLINEHELP_TESTS_MOCK_HELPTOPIC_H
#define GMX_ONLINEHELP_TESTS_MOCK_HELPTOPIC_H

#include <gmock/gmock.h>

#include "gromacs/onlinehelp/helptopic.h"
#include "gromacs/onlinehelp/helpwritercontext.h"

namespace gmx
{
namespace test
{

class MockHelpTopic : public AbstractCompositeHelpTopic
{
public:
    static MockHelpTopic& addSubTopic(gmx::AbstractCompositeHelpTopic* parent,
                                      const char*                      name,
                                      const char*                      title,
                                      const char*                      text);

    MockHelpTopic(const char* name, const char* title, const char* text);
    ~MockHelpTopic() override;

    const char* name() const override;
    const char* title() const override;

    MOCK_CONST_METHOD1(writeHelp, void(const HelpWriterContext& context));

    MockHelpTopic& addSubTopic(const char* name, const char* title, const char* text);
    using AbstractCompositeHelpTopic::addSubTopic;

    /*! \brief
     * Calls base class writeHelp() method.
     *
     * This provides the possibility for the mock to do the actual help
     * writing.
     */
    void writeHelpBase(const HelpWriterContext& context)
    {
        AbstractCompositeHelpTopic::writeHelp(context);
    }

private:
    std::string helpText() const override;

    const char* name_;
    const char* title_;
    const char* text_;
};

} // namespace test
} // namespace gmx

#endif
