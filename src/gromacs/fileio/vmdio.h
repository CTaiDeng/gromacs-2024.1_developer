/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2009- The GROMACS Authors
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

#ifndef GMX_FILEIO_VMDIO_H
#define GMX_FILEIO_VMDIO_H

#include <filesystem>

#include "external/vmd_molfile/molfile_plugin.h"

#include "gromacs/utility/basedefinitions.h"

struct t_trxframe;

struct gmx_vmdplugin_t
{
    molfile_plugin_t*     api;
    std::filesystem::path filetype;
    void*                 handle;
    gmx_bool              bV;
};

int read_first_vmd_frame(const std::filesystem::path& fn, gmx_vmdplugin_t** vmdpluginp, t_trxframe* fr);
gmx_bool read_next_vmd_frame(gmx_vmdplugin_t* vmdplugin, t_trxframe* fr);

#endif
