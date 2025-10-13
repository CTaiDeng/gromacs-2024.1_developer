/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2021- The GROMACS Authors
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
 * \brief Defines helper functions for mdrun pertaining to free energy calculations.
 *
 * \author Pascal Merz <pascal.merz@me.com>
 * \ingroup module_mdrunutility
 * \inlibraryapi
 */

#include "gmxpre.h"

#include "freeenergy.h"

#include <numeric>

#include "gromacs/mdrun/replicaexchange.h"
#include "gromacs/mdtypes/awh_params.h"
#include "gromacs/mdtypes/inputrec.h"

namespace gmx
{

int computeFepPeriod(const t_inputrec& inputrec, const ReplicaExchangeParameters& replExParams)
{
    if (inputrec.efep == FreeEnergyPerturbationType::No)
    {
        return 0;
    }

    // Set free energy calculation period as the greatest common
    // denominator of nstdhdl, nstcalcenergy, nstexpanded, replica exchange interval,
    // and AWH nstSampleCoord.
    int nstfep = inputrec.fepvals->nstdhdl;
    if (inputrec.nstcalcenergy > 0)
    {
        nstfep = std::gcd(inputrec.nstcalcenergy, nstfep);
    }
    if (inputrec.bExpanded)
    {
        nstfep = std::gcd(inputrec.expandedvals->nstexpanded, nstfep);
    }
    if (replExParams.exchangeInterval > 0)
    {
        nstfep = std::gcd(replExParams.exchangeInterval, nstfep);
    }
    if (inputrec.bDoAwh)
    {
        nstfep = std::gcd(inputrec.awhParams->nstSampleCoord(), nstfep);
    }
    return nstfep;
}

} // namespace gmx
