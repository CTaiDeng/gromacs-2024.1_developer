/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2019- The GROMACS Authors
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

/*!\internal
 * \file
 * \brief
 * Implements setatoms class.
 *
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 * \ingroup module_coordinateio
 */

#include "gmxpre.h"

#include "setatoms.h"

#include <algorithm>

#include "gromacs/coordinateio/coordinatefileenums.h"
#include "gromacs/trajectory/trajectoryframe.h"
#include "gromacs/utility/exceptions.h"

namespace gmx
{

void SetAtoms::checkAbilityDependencies(unsigned long abilities) const
{
    if ((abilities & convertFlag(moduleRequirements_)) == 0U)
    {
        std::string errorMessage =
                "Output file type does not support writing atom information. "
                "You need to use PDB, GRO or TNG as the file type for this.";
        GMX_THROW(InconsistentInputError(errorMessage.c_str()));
    }
}

void SetAtoms::processFrame(const int /*framenumber*/, t_trxframe* input)
{
    switch (atomFlag_)
    {
        case (ChangeAtomsType::Never):
            input->bAtoms = false;
            input->atoms  = nullptr;
            break;
        case (ChangeAtomsType::Always):
            if (!haveAtoms(*input))
            {
                GMX_THROW(
                        InconsistentInputError("Atoms needed by output but not "
                                               "available in input frame or topology"));
            }
            input->bAtoms = true;
            if (haveStructureFileAtoms())
            {
                input->atoms = atoms();
            }
            break;
        case (ChangeAtomsType::PreservedIfPresent): break;
        case (ChangeAtomsType::AlwaysFromStructure):
            if (!haveStructureFileAtoms())
            {
                GMX_THROW(
                        InconsistentInputError("Requested to add atoms information "
                                               "to coordinate frame when it was not available"));
            }
            input->bAtoms = true;
            input->atoms  = atoms();
            break;
        default: GMX_THROW(InconsistentInputError("Value for atom flag not understood"));
    }
}

bool SetAtoms::haveFrameAtoms(const t_trxframe& input)
{
    return input.bAtoms;
}

} // namespace gmx
