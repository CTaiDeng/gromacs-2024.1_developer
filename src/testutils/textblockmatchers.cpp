/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2015- The GROMACS Authors
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
 * Implements classes from textblockmatchers.h.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_testutils
 */
#include "gmxpre.h"

#include "testutils/textblockmatchers.h"

#include <memory>
#include <regex>

#include "gromacs/utility/stringstream.h"
#include "gromacs/utility/textreader.h"
#include "gromacs/utility/textwriter.h"

#include "testutils/refdata.h"

namespace gmx
{
namespace test
{

namespace
{

class ExactTextMatcher : public ITextBlockMatcher
{
public:
    void checkStream(TextInputStream* stream, TestReferenceChecker* checker) override
    {
        TextReader reader(stream);
        checker->checkTextBlock(reader.readAll(), "Contents");
    }
};

class NoTextMatcher : public ITextBlockMatcher
{
public:
    void checkStream(TextInputStream* /*stream*/, TestReferenceChecker* /*checker*/) override {}
};

class FilteringExactTextMatcher : public ITextBlockMatcher
{
public:
    FilteringExactTextMatcher(const std::vector<std::string>& linesToSkip,
                              const bool                      trimLeadingWhiteSpace,
                              const bool                      trimTrailingWhiteSpace) :
        trimLeadingWhiteSpace_(trimLeadingWhiteSpace), trimTrailingWhiteSpace_(trimTrailingWhiteSpace)
    {
        // Prepare the regular expressions to filter out of the stream.
        for (const auto& lineToSkip : linesToSkip)
        {
            regexesToSkip_.emplace_back(lineToSkip, std::regex::nosubs | std::regex::extended);
        }
    }

    void checkStream(TextInputStream* stream, TestReferenceChecker* checker) override
    {
        StringOutputStream filteredStream;
        {
            TextReader reader(stream);
            reader.setTrimLeadingWhiteSpace(trimLeadingWhiteSpace_);
            reader.setTrimTrailingWhiteSpace(trimTrailingWhiteSpace_);
            TextWriter writer(&filteredStream);

            // Filter the stream
            std::string line;
            while (reader.readLine(&line))
            {
                bool shouldSkip = false;
                for (const auto& regexToSkip : regexesToSkip_)
                {
                    if (std::regex_match(line.c_str(), regexToSkip))
                    {
                        shouldSkip = true;
                        break;
                    }
                }
                if (!shouldSkip)
                {
                    writer.writeLine(line);
                }
            }
        }
        StringInputStream filteredStreamToCheck(filteredStream.toString());
        TextReader        reader(&filteredStreamToCheck);
        checker->checkTextBlock(reader.readAll(), "Contents");
    }

    std::vector<std::regex> regexesToSkip_;
    const bool              trimLeadingWhiteSpace_;
    const bool              trimTrailingWhiteSpace_;
};

} // namespace

ITextBlockMatcher::~ITextBlockMatcher() {}

ITextBlockMatcherSettings::~ITextBlockMatcherSettings() {}

TextBlockMatcherPointer ExactTextMatch::createMatcher() const
{
    return TextBlockMatcherPointer(std::make_unique<ExactTextMatcher>());
}

TextBlockMatcherPointer NoTextMatch::createMatcher() const
{
    return TextBlockMatcherPointer(std::make_unique<NoTextMatcher>());
}

FilteringExactTextMatch::FilteringExactTextMatch(std::vector<std::string> linesToSkip,
                                                 const bool               trimLeadingWhiteSpace,
                                                 const bool               trimTrailingWhiteSpace) :
    linesToSkip_(std::move(linesToSkip)),
    trimLeadingWhiteSpace_(trimLeadingWhiteSpace),
    trimTrailingWhiteSpace_(trimTrailingWhiteSpace)
{
}

void FilteringExactTextMatch::addRegexToSkip(const std::string& lineToSkip)
{
    linesToSkip_.push_back(lineToSkip);
}

TextBlockMatcherPointer FilteringExactTextMatch::createMatcher() const
{
    return TextBlockMatcherPointer(std::make_unique<FilteringExactTextMatcher>(
            linesToSkip_, trimLeadingWhiteSpace_, trimTrailingWhiteSpace_));
}

} // namespace test
} // namespace gmx
