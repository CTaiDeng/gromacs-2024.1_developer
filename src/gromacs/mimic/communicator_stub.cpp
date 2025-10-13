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

#include "gromacs/math/units.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/exceptions.h"

#include "communicator.h"

namespace gmx
{

#ifdef __clang__
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wmissing-noreturn"
#endif

void MimicCommunicator::init()
{
    GMX_THROW(InternalError(
            "GROMACS is compiled without MiMiC support! Please, reconfigure with -DGMX_MIMIC=ON"));
}

void MimicCommunicator::sendInitData(gmx_mtop_t* /*mtop*/, ArrayRef<const RVec> /*coords*/)
{
    GMX_THROW(InternalError(
            "GROMACS is compiled without MiMiC support! Please, reconfigure with -DGMX_MIMIC=ON"));
}

int64_t MimicCommunicator::getStepNumber()
{
    GMX_THROW(InternalError(
            "GROMACS is compiled without MiMiC support! Please, reconfigure with -DGMX_MIMIC=ON"));
}

void MimicCommunicator::getCoords(ArrayRef<RVec> /*x*/, const int /*natoms*/)
{
    GMX_THROW(InternalError(
            "GROMACS is compiled without MiMiC support! Please, reconfigure with -DGMX_MIMIC=ON"));
}

void MimicCommunicator::sendEnergies(real /*energy*/)
{
    GMX_THROW(InternalError(
            "GROMACS is compiled without MiMiC support! Please, reconfigure with -DGMX_MIMIC=ON"));
}

void MimicCommunicator::sendForces(ArrayRef<RVec> /*forces*/, int /*natoms*/)
{
    GMX_THROW(InternalError(
            "GROMACS is compiled without MiMiC support! Please, reconfigure with -DGMX_MIMIC=ON"));
}

void MimicCommunicator::finalize()
{
    GMX_THROW(InternalError(
            "GROMACS is compiled without MiMiC support! Please, reconfigure with -DGMX_MIMIC=ON"));
}

#ifdef __clang__
#    pragma clang diagnostic pop
#endif

} // namespace gmx
