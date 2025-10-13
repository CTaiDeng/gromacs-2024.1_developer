/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2017- The GROMACS Authors
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
 * Implements routine to check the content of conf files.
 *
 * \author David van der Spoel <david.vanderspoel@icm.uu.se>
 * \ingroup module_testutils
 */
#include "gmxpre.h"

#include "testutils/conftest.h"

#include <cstdio>
#include <cstdlib>

#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/stringutil.h"
#include "gromacs/utility/textreader.h"
#include "gromacs/utility/textstream.h"

#include "testutils/refdata.h"
#include "testutils/testasserts.h"
#include "testutils/textblockmatchers.h"

namespace gmx
{

namespace test
{

namespace
{

class ConfMatcher : public ITextBlockMatcher
{
public:
    explicit ConfMatcher(const ConfMatchSettings& settings) : settings_(settings) {}

    void checkStream(TextInputStream* stream, TestReferenceChecker* checker) override
    {
        checkConfFile(stream, checker, settings_);
    }

private:
    ConfMatchSettings settings_;
};

} // namespace

void checkConfFile(TextInputStream* input, TestReferenceChecker* checker, const ConfMatchSettings& settings)
{

    TestReferenceChecker groChecker(checker->checkCompound("GroFile", "Header"));
    // Check the first two lines of the output file
    std::string line;
    EXPECT_TRUE(input->readLine(&line));
    line = stripSuffixIfPresent(line, "\n");
    groChecker.checkString(line, "Title");
    EXPECT_TRUE(input->readLine(&line));
    line = stripSuffixIfPresent(line, "\n");
    groChecker.checkInteger(std::atoi(line.c_str()), "Number of atoms");
    // Check the full configuration only if required
    if (settings.matchFullConfiguration)
    {
        TextReader reader(input);
        checker->checkTextBlock(reader.readAll(), "Configuration");
    }
}

TextBlockMatcherPointer ConfMatch::createMatcher() const
{
    return TextBlockMatcherPointer(new ConfMatcher(settings_));
}

} // namespace test
} // namespace gmx
