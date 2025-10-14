/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2010- The GROMACS Authors
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

#include "gmxpre.h"

#include "fflibutil.h"

#include <cstring>

#include <filesystem>
#include <string>
#include <vector>

#include "gromacs/utility/cstringutil.h"
#include "gromacs/utility/datafilefinder.h"
#include "gromacs/utility/directoryenumerator.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/futil.h"
#include "gromacs/utility/path.h"
#include "gromacs/utility/smalloc.h"
#include "gromacs/utility/stringutil.h"

std::filesystem::path fflib_forcefield_dir_ext()
{
    return ".ff";
}

std::filesystem::path fflib_forcefield_itp()
{
    return "forcefield.itp";
}

std::filesystem::path fflib_forcefield_doc()
{
    return "forcefield.doc";
}

std::filesystem::path fflib_filename_base(const std::filesystem::path& filename)
{
    return filename.stem();
}

std::vector<std::filesystem::path> fflib_search_file_end(const std::filesystem::path& ffdir,
                                                         const char*                  file_end,
                                                         bool                         bFatalError)
{
    try
    {
        auto ffdirFull = gmx::getLibraryFileFinder().findFile(ffdir);
        auto result = gmx::DirectoryEnumerator::enumerateFilesWithExtension(ffdirFull, file_end, true);
        if (result.empty() && bFatalError)
        {
            std::string message = gmx::formatString(
                    "Could not find any files ending on '%s' "
                    "in the force field directory '%s'",
                    file_end,
                    ffdir.u8string().c_str());
            GMX_THROW(gmx::InvalidInputError(message));
        }
        for (auto& filename : result)
        {
            filename = std::filesystem::path(ffdir).append(filename.string());
        }
        return result;
    }
    GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR
}

std::vector<gmx::DataFileInfo> fflib_enumerate_forcefields()
{
    const auto&                    dirend     = fflib_forcefield_dir_ext();
    const auto&                    filename   = fflib_forcefield_itp();
    std::vector<gmx::DataFileInfo> candidates = gmx::getLibraryFileFinder().enumerateFiles(
            gmx::DataFileOptions(dirend).throwIfNotFound(false));

    std::vector<gmx::DataFileInfo> result;
    for (const auto& candidate : candidates)
    {
        auto testPath = std::filesystem::path(candidate.dir_) / candidate.name_ / filename;
        // TODO: Consider also checking that the directory can be listed.
        if (gmx::File::exists(testPath, gmx::File::returnFalseOnError))
        {
            result.push_back(candidate);
        }
    }

    // TODO: Consider merging this into enumerateFiles(), such that the error
    // could also list the directories searched.
    if (result.empty())
    {
        std::string message = gmx::formatString(
                "No force fields found (files with name '%s' "
                "in subdirectories ending on '%s')",
                filename.u8string().c_str(),
                dirend.u8string().c_str());
        GMX_THROW(gmx::InvalidInputError(message));
    }

    return result;
}

bool fflib_fexist(const std::filesystem::path& file)
{
    return !gmx::findLibraryFile(file, true, false).empty();
}


FILE* fflib_open(const std::filesystem::path& file)
{
    std::string fileFullPath = gmx::findLibraryFile(file).u8string();
    fprintf(stderr, "Opening force field file %s\n", fileFullPath.c_str());
    return gmx_ffopen(fileFullPath, "r");
}
