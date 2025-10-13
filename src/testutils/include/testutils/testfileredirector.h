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

/*! \libinternal \file
 * \brief
 * Declares generic mock implementations for interfaces in fileredirector.h.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \inlibraryapi
 * \ingroup module_testutils
 */
#ifndef GMX_TESTUTILS_TESTFILEREDIRECTOR_H
#define GMX_TESTUTILS_TESTFILEREDIRECTOR_H

#include <memory>
#include <set>
#include <string>

#include "gromacs/utility/classhelpers.h"
#include "gromacs/utility/fileredirector.h"

namespace gmx
{
namespace test
{

class TestReferenceChecker;

/*! \libinternal \brief
 * In-memory implementation for IFileInputRedirector for tests.
 *
 * By default, this implementation will return `false` for all file existence
 * checks.  To return `true` for a specific path, use addExistingFile().
 *
 * \inlibraryapi
 * \ingroup module_testutils
 */
class TestFileInputRedirector : public IFileInputRedirector
{
public:
    TestFileInputRedirector();
    ~TestFileInputRedirector() override;

    /*! \brief
     * Marks the provided path as an existing file.
     *
     * \throws std::bad_alloc if out of memory.
     *
     * Further checks for existence of the given path will return `true`.
     */
    void addExistingFile(const std::filesystem::path& filename);

    // From IFileInputRedirector
    bool fileExists(const std::filesystem::path& filename,
                    const File::NotFoundHandler& onNotFound) const override;

private:
    std::set<std::filesystem::path> existingFiles_;

    GMX_DISALLOW_COPY_AND_ASSIGN(TestFileInputRedirector);
};

/*! \libinternal \brief
 * In-memory implementation of IFileOutputRedirector for tests.
 *
 * This class redirects all output files to in-memory buffers, and supports
 * checking the contents of these files using the reference data framework.
 *
 * \ingroup module_testutils
 */
class TestFileOutputRedirector : public IFileOutputRedirector
{
public:
    TestFileOutputRedirector();
    ~TestFileOutputRedirector() override;

    /*! \brief
     * Checks contents of all redirected files (including stdout).
     *
     * This method should not be called if the redirector will still be
     * used for further output in the test.  Behavior is not designed for
     * checking in the middle of the test, although that could potentially
     * be changed if necessary.
     */
    void checkRedirectedFiles(TestReferenceChecker* checker);

    // From IFileOutputRedirector
    TextOutputStream&       standardOutput() override;
    TextOutputStreamPointer openTextOutputFile(const std::filesystem::path& filename) override;

private:
    class Impl;

    std::unique_ptr<Impl> impl_;
};

} // namespace test
} // namespace gmx

#endif
