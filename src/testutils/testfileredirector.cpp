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

/*! \internal \file
 * \brief
 * Implements classes from testfileredirector.h.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_testutils
 */
#include "gmxpre.h"

#include "testutils/testfileredirector.h"

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "gromacs/utility/stringstream.h"

#include "testutils/stringtest.h"

namespace gmx
{
namespace test
{

/********************************************************************
 * TestFileInputRedirector
 */

TestFileInputRedirector::TestFileInputRedirector() {}

TestFileInputRedirector::~TestFileInputRedirector() {}

void TestFileInputRedirector::addExistingFile(const std::filesystem::path& filename)
{
    existingFiles_.insert(filename);
}

bool TestFileInputRedirector::fileExists(const std::filesystem::path& filename,
                                         const File::NotFoundHandler& onNotFound) const
{
    if (existingFiles_.count(filename) == 0)
    {
        File::NotFoundInfo info(filename, "File not present in test", nullptr, false, 0);
        onNotFound(info);
        return false;
    }
    return true;
}

/********************************************************************
 * TestFileOutputRedirector::Impl
 */

class TestFileOutputRedirector::Impl
{
public:
    typedef std::shared_ptr<StringOutputStream>         StringStreamPointer;
    typedef std::pair<std::string, StringStreamPointer> FileListEntry;

    StringStreamPointer        stdoutStream_;
    std::vector<FileListEntry> fileList_;
};

/********************************************************************
 * TestFileOutputRedirector
 */

TestFileOutputRedirector::TestFileOutputRedirector() : impl_(new Impl) {}

TestFileOutputRedirector::~TestFileOutputRedirector() {}

TextOutputStream& TestFileOutputRedirector::standardOutput()
{
    if (!impl_->stdoutStream_)
    {
        impl_->stdoutStream_.reset(new StringOutputStream);
        impl_->fileList_.emplace_back("<stdout>", impl_->stdoutStream_);
    }
    return *impl_->stdoutStream_;
}

TextOutputStreamPointer TestFileOutputRedirector::openTextOutputFile(const std::filesystem::path& filename)
{
    Impl::StringStreamPointer stream(new StringOutputStream);
    impl_->fileList_.emplace_back(filename.u8string(), stream);
    return stream;
}

void TestFileOutputRedirector::checkRedirectedFiles(TestReferenceChecker* checker)
{
    std::vector<Impl::FileListEntry>::const_iterator i;
    for (i = impl_->fileList_.begin(); i != impl_->fileList_.end(); ++i)
    {
        StringTestBase::checkText(checker, i->second->toString(), i->first.c_str());
    }
}

} // namespace test
} // namespace gmx
