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
 * Implements gmx::TextReader.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_utility
 */
#include "gmxpre.h"

#include "gromacs/utility/textreader.h"

#include "gromacs/utility/filestream.h"
#include "gromacs/utility/nodelete.h"
#include "gromacs/utility/textstream.h"

namespace gmx
{

// static
std::string TextReader::readFileToString(const char* filename)
{
    TextReader  reader(filename);
    std::string result(reader.readAll());
    reader.close();
    return result;
}

// static
std::string TextReader::readFileToString(const std::string& filename)
{
    return readFileToString(filename.c_str());
}

//! Implementation class
class TextReader::Impl
{
public:
    //! Constructor.
    explicit Impl(const TextInputStreamPointer& stream) :
        stream_(stream),
        trimLeadingWhiteSpace_(false),
        trimTrailingWhiteSpace_(false),
        trimTrailingComment_(false),
        commentChar_(0)
    {
    }

    //! Stream used by this reader.
    TextInputStreamPointer stream_;
    //! Whether leading whitespace should be removed.
    bool trimLeadingWhiteSpace_;
    //! Whether trailing whitespace should be removed.
    bool trimTrailingWhiteSpace_;
    //! Whether a trailing comment should be removed.
    bool trimTrailingComment_;
    /*! \brief Character that denotes the start of a comment on a line.
     *
     * Zero until TextReader::setTrimTrailingComment is called to
     * activate such trimming with a given character. */
    char commentChar_;
};

TextReader::TextReader(const std::string& filename) :
    impl_(new Impl(TextInputStreamPointer(new TextInputFile(filename))))
{
}

TextReader::TextReader(TextInputStream* stream) :
    impl_(new Impl(TextInputStreamPointer(stream, no_delete<TextInputStream>())))
{
}

TextReader::TextReader(const TextInputStreamPointer& stream) : impl_(new Impl(stream)) {}

TextReader::~TextReader() {}

bool TextReader::readLine(std::string* linePtr)
{
    if (!impl_->stream_->readLine(linePtr))
    {
        return false;
    }
    auto&      line              = *linePtr;
    const char whiteSpaceChars[] = " \t\r\n";
    if (impl_->trimLeadingWhiteSpace_)
    {
        const size_t endPos = line.find_first_not_of(whiteSpaceChars);
        if (endPos == std::string::npos)
        {
            line.resize(0);
        }
        else
        {
            line = line.substr(endPos, std::string::npos);
        }
    }
    if (impl_->trimTrailingComment_)
    {
        auto commentPos = line.find(impl_->commentChar_);
        if (commentPos != std::string::npos)
        {
            line.resize(commentPos);
        }
    }
    if (impl_->trimTrailingWhiteSpace_)
    {
        const size_t endPos = line.find_last_not_of(whiteSpaceChars);
        if (endPos == std::string::npos)
        {
            line.resize(0);
        }
        else
        {
            line.resize(endPos + 1);
        }
    }
    return true;
}

void TextReader::setTrimLeadingWhiteSpace(bool doTrimming)
{
    impl_->trimLeadingWhiteSpace_ = doTrimming;
}

void TextReader::setTrimTrailingWhiteSpace(bool doTrimming)
{
    impl_->trimTrailingWhiteSpace_ = doTrimming;
}

void TextReader::setTrimTrailingComment(bool doTrimming, char commentChar)
{
    impl_->trimTrailingComment_ = doTrimming;
    if (impl_->trimTrailingComment_)
    {
        impl_->commentChar_ = commentChar;
    }
}

std::string TextReader::readAll()
{
    std::string result;
    std::string line;
    while (readLine(&line))
    {
        result.append(line);
    }
    return result;
}

void TextReader::close()
{
    impl_->stream_->close();
}

} // namespace gmx
