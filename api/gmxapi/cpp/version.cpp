/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2018- The GROMACS Authors
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

#include "gmxpre.h"

#include "gmxapi/version.h"

namespace gmxapi
{

version_t Version::majorVersion()
{
    return c_majorVersion;
}

version_t Version::minorVersion()
{
    return c_minorVersion;
}

version_t Version::patchVersion()
{
    return c_patchVersion;
}

std::string Version::release()
{
    return c_release;
}

bool Version::hasFeature(const std::string& featurename)
{
    // For features introduced without an incompatible API change or where
    // semantic versioning is otherwise insufficient, we can consult a map, TBD.
    (void)featurename;
    return false;
}

bool Version::isAtLeast(version_t major, version_t minor, version_t patch)
{
    if (Version::majorVersion() < major)
    {
        return false;
    }
    if (Version::majorVersion() > major)
    {
        return true;
    }
    if (Version::minorVersion() < minor)
    {
        return false;
    }
    if (Version::minorVersion() > minor)
    {
        return true;
    }
    return Version::patchVersion() >= patch;
}


} // end namespace gmxapi
