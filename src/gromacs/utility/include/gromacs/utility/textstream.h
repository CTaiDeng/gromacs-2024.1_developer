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
 * Declares interfaces for simple input/output streams.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \inlibraryapi
 * \ingroup module_utility
 */
#ifndef GMX_UTILITY_TEXTSTREAM_H
#define GMX_UTILITY_TEXTSTREAM_H

#include <memory>

namespace gmx
{

/*! \libinternal \brief
 * Interface for reading text.
 *
 * Concrete implementations can read the text from, e.g., a file or an in-memory
 * string.  The main use is to allow unit tests to inject in-memory buffers
 * instead of writing files to be read by the code under test, but there are
 * also use cases outside the tests where it is useful to abstract out whether
 * the input is from a real file or something else.
 *
 * To use more advanced formatting than reading raw lines, use TextReader.
 *
 * Both methods in the interface can throw std::bad_alloc or other exceptions
 * that indicate failures to read from the stream.
 *
 * \inlibraryapi
 * \ingroup module_utility
 */
class TextInputStream
{
public:
    virtual ~TextInputStream() {}

    /*! \brief
     * Reads a line (with newline included) from the stream.
     *
     * \param[out] line    String to receive the line.
     * \returns    `false` if nothing was read because the stream ended.
     *
     * On error or when `false` is returned, \p line will be empty.
     */
    virtual bool readLine(std::string* line) = 0;
    /*! \brief
     * Closes the stream.
     *
     * It is not allowed to read from a stream after it has been closed.
     * See TextOutputStream::close() for rationale for a close() method
     * separate from the destructor.  For input, failures during close
     * should be rare, but it is clearer to keep the interface symmetric.
     */
    virtual void close() = 0;
};

/*! \libinternal \brief
 * Interface for writing text.
 *
 * Concrete implementations can write the text to, e.g., a file or an in-memory
 * string.  The main use is to allow unit tests to inject in-memory buffers
 * instead of reading in files produced by the code under test, but there are
 * also use cases outside the tests where it is useful to abstract out whether
 * the output is into a real file or something else.
 *
 * To use more advanced formatting than writing plain strings, use TextWriter.
 *
 * The current implementation assumes text-only output in several places, but
 * this interface could possibly be generalized also for binary files.
 * However, since all binary files currently written by \Gromacs are either
 * XDR- or TNG-based, they may require a different approach.  Also, it is worth
 * keeping the distinction between text and binary files clear, since Windows
 * does transparent `LF`-`CRLF` newline translation for text files, so mixing
 * modes when reading and/or writing the same file can cause subtle issues.
 *
 * Both methods in the interface can throw std::bad_alloc or other exceptions
 * that indicate failures to write to the stream.
 *
 * \inlibraryapi
 * \ingroup module_utility
 */
class TextOutputStream
{
public:
    virtual ~TextOutputStream() {}

    /*! \brief
     * Writes a given string to the stream.
     */
    virtual void write(const char* text) = 0;
    /*! \brief
     * Closes the stream.
     *
     * It is not allowed to write to a stream after it has been closed.
     * A method separate from the destructor is provided such that errors
     * that occur while closing the stream (e.g., when closing the file)
     * can be handled using exceptions.
     * The destructor is not allowed to throw, so code that wants to
     * observe such errors needs to call close() after it has finished
     * writing to the stream.
     */
    virtual void close() = 0;
};

//! Shorthand for a smart pointer to a TextInputStream.
typedef std::shared_ptr<TextInputStream> TextInputStreamPointer;
//! Shorthand for a smart pointer to a TextOutputStream.
typedef std::shared_ptr<TextOutputStream> TextOutputStreamPointer;

} // namespace gmx

#endif
