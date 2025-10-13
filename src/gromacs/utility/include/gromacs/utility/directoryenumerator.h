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
 * Declares gmx::DirectoryEnumerator.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \inlibraryapi
 * \ingroup module_utility
 */
#ifndef GMX_UTILITY_DIRECTORYENUMERATOR_H
#define GMX_UTILITY_DIRECTORYENUMERATOR_H

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace gmx
{

/*! \libinternal \brief
 * Lists files in a directory.
 *
 * If multiple threads share the same DirectoryEnumerator, they must
 * take responsibility for their mutual synchronization, particularly
 * with regard to calling nextFile().
 *
 * \inlibraryapi
 * \ingroup module_utility
 */
class DirectoryEnumerator
{
public:
    /*! \brief
     * Convenience function to list files with certain extension from a
     * directory.
     *
     * \param[in]  dirname   Path to the directory to list.
     * \param[in]  extension List files with the given extension
     *     (or suffix in file name).
     * \param[in]  bThrow    Whether failure to open the directory should throw.
     * \returns    List of files with the given extension in \p dirname.
     * \throws std::bad_alloc if out of memory.
     * \throws FileIOError if opening the directory fails and `bThrow == true`.
     * \throws FileIOError if some other I/O error occurs.
     */
    static std::vector<std::filesystem::path> enumerateFilesWithExtension(const std::filesystem::path& dirname,
                                                                          const std::string& extension,
                                                                          bool bThrow);

    /*! \brief
     * Opens a directory for listing.
     *
     * \param[in] dirname Path to the directory to list.
     * \param[in] bThrow  Whether failure to open the directory should throw.
     * \throws std::bad_alloc if out of memory.
     * \throws FileIOError if opening the directory fails and `bThrow == true`
     */
    explicit DirectoryEnumerator(const std::filesystem::path& dirname, bool bThrow = true);
    ~DirectoryEnumerator();

    /*! \brief
     * Gets next file in a directory.
     *
     * \returns Optional name of next file in directory.
     * \throws  std::bad_alloc if out of memory.
     * \throws  FileIOError if listing the next file fails.
     *
     * If all files from the directory have been returned (or there are no
     * files in the directory and this is the first call), the method
     * returns std::nullopt.
     * Otherwise, the return value is the optional filename without path information.
     *
     * If `bThrow` passed to the constructor was `false` and the directory
     * was not successfully opened, the first call to this function will
     * return `false`.
     *
     * This method is not thread safe when called on the same
     * object by multiple threads. Such use requires external
     * synchronization.
     */
    std::optional<std::filesystem::path> nextFile();

private:
    class Impl;

    std::unique_ptr<Impl> impl_;
};

} // namespace gmx

#endif
