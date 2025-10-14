/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2023- The GROMACS Authors
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
 * Implements the Colvars GROMACS proxy class during pre-processing.
 *
 * \author Hubert Santuz <hubert.santuz@gmail.com>
 * \ingroup module_applied_forces
 */
#include "gmxpre.h"

#include "colvarspreprocessor.h"

#include <string>

namespace gmx
{

ColvarsPreProcessor::ColvarsPreProcessor(const std::string&   colvarsConfigString,
                                         t_atoms              atoms,
                                         PbcType              pbcType,
                                         const MDLogger*      logger,
                                         real                 ensembleTemperature,
                                         int                  seed,
                                         const matrix         box,
                                         ArrayRef<const RVec> x) :
    ColvarProxyGromacs(colvarsConfigString,
                       atoms,
                       pbcType,
                       logger,
                       true,
                       std::map<std::string, std::string>(),
                       ensembleTemperature,
                       seed),
    x_(x)
{

    // Initialize t_pbc struct
    set_pbc(&gmxPbc_, pbcType, box);

    cvm::log(cvm::line_marker);
    cvm::log("End colvars Initialization.\n\n");
}

std::vector<RVec> ColvarsPreProcessor::getColvarsCoords()
{

    std::vector<RVec> colvarsCoords;

    for (const auto& atom_id : atoms_ids)
    {
        colvarsCoords.push_back(x_[atom_id]);
    }
    return colvarsCoords;
}

bool ColvarsPreProcessor::inputStreamsToKVT(KeyValueTreeObjectBuilder treeBuilder, const std::string& tag)
{

    // Save full copy of the content of the input streams (aka input files) into the KVT.
    for (const auto& inputName : list_input_stream_names())
    {
        std::istream&      stream = input_stream(inputName);
        std::ostringstream os;
        os << stream.rdbuf();
        std::string key = tag;
        key += "-";
        key += inputName;
        treeBuilder.addValue<std::string>(key, os.str());
    }
    return true;
}


} // namespace gmx
