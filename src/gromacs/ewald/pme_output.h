/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2020- The GROMACS Authors
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

/*! \libinternal \file
 * \brief Defines a struct useful for transferring the PME output
 * values
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \ingroup module_ewald
 */

#ifndef GMX_EWALD_PME_OUTPUT_H
#define GMX_EWALD_PME_OUTPUT_H

#include "gromacs/math/vectypes.h"
#include "gromacs/utility/arrayref.h"

// TODO There's little value in computing the Coulomb and LJ virial
// separately, so we should simplify that.
// TODO The matrices might be best as a view, but not currently
// possible. Use mdspan?
struct PmeOutput
{
    //!< Host staging area for PME forces
    gmx::ArrayRef<gmx::RVec> forces_;
    //!< True if forces have been staged other false (when forces are reduced on the GPU).
    bool haveForceOutput_ = false;
    //!< Host staging area for PME coulomb energy
    real coulombEnergy_ = 0;
    //!< Host staging area for PME coulomb virial contributions
    matrix coulombVirial_ = { { 0 } };
    //!< Host staging area for PME coulomb dVdl.
    real coulombDvdl_ = 0;
    //!< Host staging area for PME LJ dVdl.
    real lennardJonesDvdl_ = 0;
    //!< Host staging area for PME LJ energy
    real lennardJonesEnergy_ = 0;
    //!< Host staging area for PME LJ virial contributions
    matrix lennardJonesVirial_ = { { 0 } };
};

#endif
