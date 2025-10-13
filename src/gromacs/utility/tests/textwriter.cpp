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
 * Tests for gmx::TextWriter.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_utility
 */
#include "gmxpre.h"

#include "gromacs/utility/textwriter.h"

#include <string>

#include <gtest/gtest.h>

#include "gromacs/utility/stringstream.h"
#include "gromacs/utility/stringutil.h"

#include "testutils/stringtest.h"

namespace
{

class TextWriterTest : public gmx::test::StringTestBase
{
public:
    TextWriterTest() : writer_(&stream_) {}

    void checkOutput() { checkText(stream_.toString(), "Output"); }

    gmx::StringOutputStream stream_;
    gmx::TextWriter         writer_;
};

TEST_F(TextWriterTest, WritesLines)
{
    writer_.writeLine("Explicit newline\n");
    writer_.writeLine("Implicit newline");
    writer_.writeLine(std::string("Explicit newline\n"));
    writer_.writeLine(std::string("Implicit newline"));
    writer_.writeLine();
    checkOutput();
}

TEST_F(TextWriterTest, WritesLinesInParts)
{
    writer_.writeString("Partial ");
    writer_.writeString("spaced");
    writer_.writeString(" line");
    writer_.writeLine();
    writer_.writeString(std::string("Partial "));
    writer_.writeString(std::string("spaced"));
    writer_.writeString(std::string(" line"));
    writer_.writeLine();
    checkOutput();
}

TEST_F(TextWriterTest, WritesWrappedLines)
{
    writer_.wrapperSettings().setIndent(2);
    writer_.wrapperSettings().setLineLength(15);
    writer_.writeLine("Wrapped and indented text");
    writer_.writeLine(std::string("Wrapped and indented text"));
    writer_.writeLine();
    checkOutput();
}

TEST_F(TextWriterTest, WritesLinesInPartsWithWrapper)
{
    writer_.wrapperSettings().setLineLength(50);
    writer_.writeString("Partial ");
    writer_.writeString("spaced");
    writer_.writeString(" line");
    writer_.writeLine();
    writer_.writeString(std::string("Partial "));
    writer_.writeString(std::string("spaced"));
    writer_.writeString(std::string(" line"));
    writer_.writeLine();
    checkOutput();
}

TEST_F(TextWriterTest, TracksNewlines)
{
    writer_.ensureLineBreak();
    writer_.ensureEmptyLine();
    writer_.writeString("First line");
    writer_.ensureLineBreak();
    writer_.ensureLineBreak();
    writer_.writeString("Second line");
    writer_.ensureEmptyLine();
    writer_.writeLine("Third line");
    writer_.ensureEmptyLine();
    writer_.writeString(std::string("Fourth line\n"));
    writer_.ensureLineBreak();
    writer_.writeString(std::string("Fifth line\n\n"));
    writer_.ensureEmptyLine();
    writer_.writeString(std::string("Sixth line"));
    writer_.ensureEmptyLine();
    checkOutput();
}

TEST_F(TextWriterTest, PreservesTrailingWhitespace)
{
    writer_.writeString("Line   ");
    writer_.writeLine();
    writer_.writeString(std::string("Line   "));
    writer_.writeLine();
    writer_.writeLine("Line   ");
    writer_.writeLine(std::string("Line   "));
    writer_.writeString("Line   \n");
    writer_.writeString(std::string("Line   \n"));
    checkOutput();
}

} // namespace
