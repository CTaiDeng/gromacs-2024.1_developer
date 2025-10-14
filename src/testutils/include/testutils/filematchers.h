/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2017- The GROMACS Authors
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
 * Declares utility classes for testing file contents.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \inlibraryapi
 * \ingroup module_testutils
 */
#ifndef GMX_TESTUTILS_FILEMATCHERS_H
#define GMX_TESTUTILS_FILEMATCHERS_H

#include <memory>
#include <string>

namespace gmx
{
namespace test
{

class ITextBlockMatcherSettings;
class TestReferenceChecker;

/*! \libinternal \brief
 * Represents a file matcher, matching file contents against reference (or
 * other) data.
 *
 * Typical pattern of declaring such matchers is to
 *  - Create a factory that implements IFileMatcherSettings,
 *  - Make that factory provide any necessary parameters that the matcher needs,
 *    using a "named parameter" idiom (see XvgMatch for an example), and
 *  - Make the factory create and return an instance of an internal
 *    implementation class that implements IFileMatcher and provides
 *    the actual matching logic.
 *
 * Any method that then wants to accept a matcher can accept a
 * IFileMatcherSettings.
 *
 * \inlibraryapi
 * \ingroup module_testutils
 */
class IFileMatcher
{
public:
    virtual ~IFileMatcher();

    /*! \brief
     * Matches contents of a file.
     *
     * \param  path     Path to the file to match.
     * \param  checker  Checker to use for matching.
     *
     * The method can change the state of the provided checker (e.g., by
     * changing the default tolerance).
     * The caller is responsible of providing a checker where such state
     * changes do not matter.
     */
    virtual void checkFile(const std::string& path, TestReferenceChecker* checker) = 0;
};

//! Smart pointer for managing a IFileMatcher.
typedef std::unique_ptr<IFileMatcher> FileMatcherPointer;

/*! \libinternal \brief
 * Represents a factory for creating a file matcher.
 *
 * See derived classes for available matchers.  Each derived class represents
 * one type of matcher (see IFileMatcher), and provides any methods
 * necessary to pass parameters to such a matcher.  Methods that accept a
 * matcher can then take in this interface, and call createFileMatcher() to use
 * the matcher that the caller of the method specifies.
 *
 * \inlibraryapi
 * \ingroup module_testutils
 */
class IFileMatcherSettings
{
public:
    //! Factory method that constructs the matcher after parameters are set.
    virtual FileMatcherPointer createFileMatcher() const = 0;

protected:
    virtual ~IFileMatcherSettings();
};

/*! \libinternal \brief
 * Use a ITextBlockMatcher for matching the contents.
 *
 * \inlibraryapi
 * \ingroup module_testutils
 */
class TextFileMatch : public IFileMatcherSettings
{
public:
    //! Creates a matcher to match contents with given text matcher.
    explicit TextFileMatch(const ITextBlockMatcherSettings& streamSettings) :
        streamSettings_(streamSettings)
    {
    }

    FileMatcherPointer createFileMatcher() const override;

private:
    const ITextBlockMatcherSettings& streamSettings_;
};

/*! \libinternal \brief
 * Do not check the contents of the file.
 *
 * \inlibraryapi
 * \ingroup module_testutils
 */
class NoContentsMatch : public IFileMatcherSettings
{
public:
    FileMatcherPointer createFileMatcher() const override;
};

} // namespace test
} // namespace gmx

#endif
