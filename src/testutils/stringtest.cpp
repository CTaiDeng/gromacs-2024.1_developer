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

/*! \internal \file
 * \brief
 * Implements gmx::test::StringTestBase.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_testutils
 */
#include "gmxpre.h"

#include "testutils/stringtest.h"

#include <string>

#include "gromacs/options/basicoptions.h"
#include "gromacs/options/ioptionscontainer.h"
#include "gromacs/utility/textreader.h"

#include "testutils/refdata.h"
#include "testutils/testoptions.h"

namespace gmx
{
namespace test
{

namespace
{
//! Stores the -stdout flag value to print out values instead of checking them.
bool g_bWriteToStdOut = false;
} // namespace

// TODO: Only add this option to those test binaries that actually need it
// (depending on the linker, it may or may not appear right now),
// or replace by a generic mechanism in TestReferenceData.
//! \cond
GMX_TEST_OPTIONS(StringTestOptions, options)
{
    options->addOption(
            BooleanOption("stdout").store(&g_bWriteToStdOut).description("Print the test string to stdout instead of checking against reference data"));
}
//! \endcond

/********************************************************************
 * StringTestBase::Impl
 */

class StringTestBase::Impl
{
public:
    TestReferenceData    data_;
    TestReferenceChecker checker_;
};

/********************************************************************
 * StringTestBase
 */

// static
void StringTestBase::checkText(TestReferenceChecker* checker, const std::string& text, const char* id)
{
    if (g_bWriteToStdOut)
    {
        printf("%s:\n", id);
        printf("%s[END]\n", text.c_str());
    }
    else
    {
        checker->checkTextBlock(text, id);
    }
}

StringTestBase::StringTestBase() : impl_(new Impl) {}

StringTestBase::~StringTestBase() {}

TestReferenceChecker& StringTestBase::checker()
{
    if (!impl_->checker_)
    {
        impl_->checker_ = impl_->data_.rootChecker();
    }
    return impl_->checker_;
}

void StringTestBase::checkText(const std::string& text, const char* id)
{
    checkText(&checker(), text, id);
}

void StringTestBase::checkFileContents(const std::string& filename, const char* id)
{
    const std::string text = TextReader::readFileToString(filename);
    checkText(text, id);
}

void StringTestBase::testFilesEqual(const std::string& refFilename, const std::string& testFilename)
{
    const std::string expectedContents = TextReader::readFileToString(refFilename);
    const std::string contents         = TextReader::readFileToString(testFilename);
    if (g_bWriteToStdOut)
    {
        printf("%s[END]\n", contents.c_str());
    }
    EXPECT_EQ(expectedContents, contents);
}

} // namespace test
} // namespace gmx
