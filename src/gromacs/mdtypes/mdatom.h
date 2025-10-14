/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 1991- The GROMACS Authors
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

/*! \brief
 * Declares mdatom data structure.
 *
 * \inpublicapi
 * \ingroup module_mdtypes
 */
#ifndef GMX_MDTYPES_MDATOM_H
#define GMX_MDTYPES_MDATOM_H

#include <vector>

#include "gromacs/math/paddedvector.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/utility/booltype.h"
#include "gromacs/utility/real.h"

enum class ParticleType : int;

typedef struct t_mdatoms
{
    //! Total mass in state A
    real tmassA;
    //! Total mass in state B
    real tmassB;
    //! Total mass
    real tmass;
    //! Number of atoms in arrays
    int nr;
    //! Number of energy groups
    int nenergrp;
    //! Do we have multiple center of mass motion removal groups
    bool bVCMgrps;
    //! Do we have any virtual sites?
    bool haveVsites;
    //! Do we have atoms that are frozen along 1 or 2 (not 3) dimensions?
    bool havePartiallyFrozenAtoms;
    //! Number of perturbed atoms
    int nPerturbed;
    //! Number of atoms for which the mass is perturbed
    int nMassPerturbed;
    //! Number of atoms for which the charge is perturbed
    int nChargePerturbed;
    //! Number of atoms for which the type is perturbed
    int nTypePerturbed;
    //! Do we have orientation restraints
    bool bOrires;
    //! Atomic mass in A state
    std::vector<real> massA;
    //! Atomic mass in B state
    std::vector<real> massB;
    //! Atomic mass in present state
    std::vector<real> massT;
    //! Inverse atomic mass per atom, 0 for vsites and shells
    gmx::PaddedVector<real> invmass;
    //! Inverse atomic mass per atom and dimension, 0 for vsites, shells and frozen dimensions
    std::vector<gmx::RVec> invMassPerDim;
    //! Atomic charge in A state
    gmx::ArrayRef<real> chargeA;
    //! Atomic charge in B state
    gmx::ArrayRef<real> chargeB;
    //! Dispersion constant C6 in A state
    std::vector<real> sqrt_c6A;
    //! Dispersion constant C6 in A state
    std::vector<real> sqrt_c6B;
    //! Van der Waals radius sigma in the A state
    std::vector<real> sigmaA;
    //! Van der Waals radius sigma in the B state
    std::vector<real> sigmaB;
    //! Van der Waals radius sigma^3 in the A state
    std::vector<real> sigma3A;
    //! Van der Waals radius sigma^3 in the B state
    std::vector<real> sigma3B;
    //! Is this atom perturbed?
    std::vector<gmx::BoolType> bPerturbed;
    //! Type of atom in the A state
    std::vector<int> typeA;
    //! Type of atom in the B state
    std::vector<int> typeB;
    //! Particle type
    std::vector<ParticleType> ptype;
    //! Group index for temperature coupling
    std::vector<unsigned short> cTC;
    //! Group index for energy matrix
    std::vector<unsigned short> cENER;
    //! Group index for acceleration
    std::vector<unsigned short> cACC;
    //! Group index for freezing
    std::vector<unsigned short> cFREEZE;
    //! Group index for center of mass motion removal
    std::vector<unsigned short> cVCM;
    //! Group index for user 1
    std::vector<unsigned short> cU1;
    //! Group index for user 2
    std::vector<unsigned short> cU2;
    //! Group index for orientation restraints
    std::vector<unsigned short> cORF;
    //! Number of atoms on this processor. TODO is this still used?
    int homenr;
    //! The lambda value used to create the contents of the struct
    real lambda;
} t_mdatoms;

#endif
