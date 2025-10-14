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

#include "gmxpre.h"

#include "cuda_version_information.h"

#include "gromacs/utility/stringutil.h"

namespace gmx
{

std::string getCudaDriverVersionString()
{
    int cuda_driver = 0;
    if (cudaDriverGetVersion(&cuda_driver) != cudaSuccess)
    {
        return "N/A";
    }
    return formatString("%d.%d", cuda_driver / 1000, cuda_driver % 100);
}

std::string getCudaRuntimeVersionString()
{
    int cuda_runtime = 0;
    if (cudaRuntimeGetVersion(&cuda_runtime) != cudaSuccess)
    {
        return "N/A";
    }
    return formatString("%d.%d", cuda_runtime / 1000, cuda_runtime % 100);
}

} // namespace gmx
