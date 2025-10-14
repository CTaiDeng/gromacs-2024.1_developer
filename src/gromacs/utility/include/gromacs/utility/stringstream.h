/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2015- The GROMACS Authors
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
 * Declares implementations for textstream.h interfaces for input/output to
 * in-memory strings.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \inlibraryapi
 * \ingroup module_utility
 */
#ifndef GMX_UTILITY_STRINGSTREAM_H
#define GMX_UTILITY_STRINGSTREAM_H

#include <string>
#include <vector>

#include "gromacs/utility/textstream.h"

namespace gmx
{

/*! \libinternal \brief
 * Text output stream implementation for writing to an in-memory string.
 *
 * Implementations for the TextOutputStream methods throw std::bad_alloc if
 * reallocation of the string fails.
 *
 * \inlibraryapi
 * \ingroup module_utility
 */
class StringOutputStream : public TextOutputStream
{
public:
    //! Returns the text written to the stream so far.
    const std::string& toString() const { return str_; }

    // From TextOutputStream
    void write(const char* text) override;
    void close() override;

private:
    std::string str_;
};

template<typename T>
class ArrayRef;

/*! \libinternal \brief
 * Helper class to convert static string data to a stream.
 *
 * Provides a text input stream that returns lines from a string
 */
class StringInputStream : public TextInputStream
{
public:
    /*! \brief
     * Constructor that stores input lines in a string.
     *
     * The string is internally but no processing is done.
     *
     * \param[in] input String to be served by the stream.
     */
    explicit StringInputStream(const std::string& input);
    /*! \brief
     * Constructor that stores input lines in a string.
     *
     * The vector of strings is stored as a string separated by newline.
     *
     * \param[in] input String to be served by the stream.
     */
    explicit StringInputStream(const std::vector<std::string>& input);
    /*! \brief
     * Constructor that stores input lines in a string.
     *
     * The array of char * is stored as a string separated by newline.
     *
     * \param[in] input Array of char * to be served by the stream.
     */
    explicit StringInputStream(ArrayRef<const char* const> const& input);

    // From TextInputStream
    bool readLine(std::string* line) override;
    void close() override {}

private:
    std::string input_;
    size_t      pos_;
};

} // namespace gmx

#endif
