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

/*! \libinternal \file
 * \brief
 * Declares gmx::test::StringTestBase.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \inlibraryapi
 * \ingroup module_testutils
 */
#ifndef GMX_TESTUTILS_STRINGTEST_H
#define GMX_TESTUTILS_STRINGTEST_H

#include <memory>
#include <string>

#include <gtest/gtest.h>

namespace gmx
{

namespace test
{

class TestReferenceChecker;

/*! \libinternal \brief
 * Test fixture for tests that check string formatting.
 *
 * For development, tests that use this fixture as their base can be run with a
 * '-stdout' command-line option to print out the tested strings to stdout.
 * If this flag is not given, they check the strings using the XML reference
 * framework (see TestReferenceData).
 *
 * Tests that need simple checking of a string, or container of
 * strings, should consider the normal implementation in
 * TestReferenceChecker.
 *
 * \inlibraryapi
 * \ingroup module_testutils
 */
class StringTestBase : public ::testing::Test
{
public:
    /*! \brief
     * Checks a block of text.
     *
     * This static method is provided for code that does not derive from
     * StringTestBase to use the same functionality, e.g., implementing the
     * `-stdout` option.
     */
    static void checkText(TestReferenceChecker* checker, const std::string& text, const char* id);

    StringTestBase();
    ~StringTestBase() override;

    /*! \brief
     * Returns the root checker for this test's reference data.
     *
     * Can be used to perform custom checks against reference data (e.g.,
     * if the test needs to check some other values than plain strings.
     */
    TestReferenceChecker& checker();

    /*! \brief
     * Checks a string.
     *
     * \param[in] text  String to check.
     * \param[in] id    Unique (within a single test) id for the string.
     */
    void checkText(const std::string& text, const char* id);
    /*! \brief
     * Checks contents of a file as a single string.
     *
     * \param[in] filename  Name of the file to check.
     * \param[in] id        Unique (within a single test) id for the string.
     *
     * Provided for convenience.  Reads the contents of \p filename into a
     * single string and calls checkText().
     */
    void checkFileContents(const std::string& filename, const char* id);

    /*! \brief
     * Tests that contents of two files are equal.
     *
     * \param[in] refFilename   File with the expected contents.
     * \param[in] testFilename  File with the contents to be tested.
     */
    static void testFilesEqual(const std::string& refFilename, const std::string& testFilename);

private:
    class Impl;

    std::unique_ptr<Impl> impl_;
};

} // namespace test
} // namespace gmx

#endif
