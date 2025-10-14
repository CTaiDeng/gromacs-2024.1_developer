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
 * Implements classes and functions from fileredirector.h.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_utility
 */
#include "gmxpre.h"

#include "gromacs/utility/fileredirector.h"

#include "gromacs/utility/filestream.h"
#include "gromacs/utility/path.h"

namespace gmx
{

IFileInputRedirector::~IFileInputRedirector() {}

IFileOutputRedirector::~IFileOutputRedirector() {}

namespace
{

/*! \internal
 * \brief
 * Implements the redirector returned by defaultFileInputRedirector().
 *
 * Does not redirect anything, but uses the file system as requested.
 *
 * \ingroup module_utility
 */
class DefaultInputRedirector : public IFileInputRedirector
{
public:
    bool fileExists(const std::filesystem::path& filename, const File::NotFoundHandler& onNotFound) const override
    {
        return File::exists(filename, onNotFound);
    }
};

/*! \internal
 * \brief
 * Implements the redirector returned by defaultFileOutputRedirector().
 *
 * Does not redirect anything, but instead opens the files exactly as
 * requested.
 *
 * \ingroup module_utility
 */
class DefaultOutputRedirector : public IFileOutputRedirector
{
public:
    TextOutputStream&       standardOutput() override { return TextOutputFile::standardOutput(); }
    TextOutputStreamPointer openTextOutputFile(const std::filesystem::path& filename) override
    {
        return TextOutputStreamPointer(new TextOutputFile(filename));
    }
};

} // namespace

//! \cond libapi
IFileInputRedirector& defaultFileInputRedirector()
{
    static DefaultInputRedirector instance;
    return instance;
}

IFileOutputRedirector& defaultFileOutputRedirector()
{
    static DefaultOutputRedirector instance;
    return instance;
}
//! \endcond

} // namespace gmx
