/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2021- The GROMACS Authors
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
 * Implements test of topology sorting routines
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \ingroup module_topology
 */
#include "gmxpre.h"

#include "gromacs/topology/topsort.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "gromacs/mdtypes/atominfo.h"
#include "gromacs/topology/idef.h"
#include "gromacs/topology/ifunc.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/arrayref.h"

namespace gmx
{
namespace
{

using testing::Eq;
using testing::IsEmpty;
using testing::Pointwise;

TEST(TopSortTest, WorksOnEmptyIdef)
{
    gmx_ffparams_t          forcefieldParams;
    InteractionDefinitions  idef(forcefieldParams);
    ArrayRef<const int64_t> emptyAtomInfo;

    gmx_sort_ilist_fe(&idef, emptyAtomInfo);

    EXPECT_EQ(0, idef.numNonperturbedInteractions[F_BONDS]) << "empty idef has no perturbed bonds";
    EXPECT_THAT(idef.il[F_BONDS].iatoms, IsEmpty());
    EXPECT_THAT(idef.il[F_LJ14].iatoms, IsEmpty());
}

//! Helper function
t_iparams makeUnperturbedBondParams(real rA, real kA)
{
    t_iparams params;
    params.harmonic = { rA, kA, rA, kA };
    return params;
}

//! Helper function
t_iparams makePerturbedBondParams(real rA, real kA, real rB, real kB)
{
    t_iparams params;
    params.harmonic = { rA, kA, rB, kB };
    return params;
}

//! Helper function
t_iparams makeUnperturbedLJ14Params(real c6A, real c12A)
{
    t_iparams params;
    params.lj14 = { c6A, c12A, c6A, c12A };
    return params;
}

//! Helper function
t_iparams makePerturbedLJ14Params(real c6A, real c12A, real c6B, real c12B)
{
    t_iparams params;
    params.lj14 = { c6A, c12A, c6B, c12B };
    return params;
}

TEST(TopSortTest, WorksOnIdefWithNoPerturbedInteraction)
{
    gmx_ffparams_t         forcefieldParams;
    InteractionDefinitions idef(forcefieldParams);

    // F_BONDS
    std::array<int, 2> bondAtoms = { 0, 1 };
    forcefieldParams.iparams.push_back(makeUnperturbedBondParams(0.9, 1000));
    idef.il[F_BONDS].push_back(forcefieldParams.iparams.size() - 1, bondAtoms);

    // F_LJ14
    forcefieldParams.iparams.push_back(makeUnperturbedLJ14Params(100, 10000));
    idef.il[F_LJ14].push_back(forcefieldParams.iparams.size() - 1, bondAtoms);
    std::vector<int64_t> atomInfo{ 0, 0 };

    gmx_sort_ilist_fe(&idef, atomInfo);

    EXPECT_EQ(1, idef.numNonperturbedInteractions[F_BONDS] / (NRAL(F_BONDS) + 1))
            << "idef with no perturbed bonds has no perturbed bonds";
    EXPECT_EQ(1, idef.numNonperturbedInteractions[F_LJ14] / (NRAL(F_LJ14) + 1))
            << "idef with no perturbed LJ 1-4 has no perturbed LJ 1-4";
    EXPECT_THAT(idef.il[F_BONDS].iatoms, Pointwise(Eq(), { 0, 0, 1 }));
    EXPECT_THAT(idef.il[F_LJ14].iatoms, Pointwise(Eq(), { 1, 0, 1 }));
}

TEST(TopSortTest, WorksOnIdefWithPerturbedInteractions)
{
    // Push the perturbed interactions last, and observe that they get
    // "sorted" to the end of the interaction lists
    gmx_ffparams_t         forcefieldParams;
    InteractionDefinitions idef(forcefieldParams);

    // F_BONDS
    std::array<int, 2> bondAtoms = { 0, 1 };
    forcefieldParams.iparams.push_back(makeUnperturbedBondParams(0.9, 1000));
    idef.il[F_BONDS].push_back(forcefieldParams.iparams.size() - 1, bondAtoms);
    std::array<int, 2> perturbedBondAtoms = { 2, 3 };
    forcefieldParams.iparams.push_back(makePerturbedBondParams(0.9, 1000, 1.1, 4000));
    idef.il[F_BONDS].push_back(forcefieldParams.iparams.size() - 1, perturbedBondAtoms);

    // F_LJ14
    forcefieldParams.iparams.push_back(makeUnperturbedLJ14Params(100, 10000));
    idef.il[F_LJ14].push_back(forcefieldParams.iparams.size() - 1, bondAtoms);
    idef.il[F_LJ14].push_back(forcefieldParams.iparams.size() - 1, perturbedBondAtoms);
    forcefieldParams.iparams.push_back(makePerturbedLJ14Params(100, 10000, 200, 20000));
    idef.il[F_LJ14].push_back(forcefieldParams.iparams.size() - 1, perturbedBondAtoms);
    // Perturb the charge of atom 2, affecting the non-perturbed LJ14 above
    std::vector<int64_t> atomInfo{ 0, 0, sc_atomInfo_HasPerturbedCharge, 0 };

    gmx_sort_ilist_fe(&idef, atomInfo);

    EXPECT_EQ(1, idef.numNonperturbedInteractions[F_BONDS] / (NRAL(F_BONDS) + 1))
            << "idef with a perturbed bond has a perturbed bond";
    EXPECT_EQ(1, idef.numNonperturbedInteractions[F_LJ14] / (NRAL(F_LJ14) + 1))
            << "idef with perturbed LJ 1-4 has perturbed LJ 1-4";
    EXPECT_THAT(idef.il[F_BONDS].iatoms, Pointwise(Eq(), { 0, 0, 1, 1, 2, 3 }));
    EXPECT_THAT(idef.il[F_LJ14].iatoms, Pointwise(Eq(), { 2, 0, 1, 2, 2, 3, 3, 2, 3 }));
}

TEST(TopSortTest, SortsIdefWithPerturbedInteractions)
{
    // Push the perturbed interactions first, and observe that they
    // get sorted to the end of the interaction lists
    gmx_ffparams_t         forcefieldParams;
    InteractionDefinitions idef(forcefieldParams);

    // F_BONDS
    std::array<int, 2> perturbedBondAtoms = { 2, 3 };
    forcefieldParams.iparams.push_back(makePerturbedBondParams(0.9, 1000, 1.1, 4000));
    idef.il[F_BONDS].push_back(forcefieldParams.iparams.size() - 1, perturbedBondAtoms);
    std::array<int, 2> bondAtoms = { 0, 1 };
    forcefieldParams.iparams.push_back(makeUnperturbedBondParams(0.9, 1000));
    idef.il[F_BONDS].push_back(forcefieldParams.iparams.size() - 1, bondAtoms);

    // F_LJ14
    forcefieldParams.iparams.push_back(makeUnperturbedLJ14Params(100, 10000));
    idef.il[F_LJ14].push_back(forcefieldParams.iparams.size() - 1, perturbedBondAtoms);
    forcefieldParams.iparams.push_back(makePerturbedLJ14Params(100, 10000, 200, 20000));
    idef.il[F_LJ14].push_back(forcefieldParams.iparams.size() - 1, perturbedBondAtoms);
    forcefieldParams.iparams.push_back(makeUnperturbedLJ14Params(100, 10000));
    idef.il[F_LJ14].push_back(forcefieldParams.iparams.size() - 1, bondAtoms);
    // Perturb the charge of atom 2, affecting the non-perturbed LJ14 above
    std::vector<int64_t> atomInfo{ 0, 0, sc_atomInfo_HasPerturbedCharge, 0 };

    gmx_sort_ilist_fe(&idef, atomInfo);

    EXPECT_EQ(1, idef.numNonperturbedInteractions[F_BONDS] / (NRAL(F_BONDS) + 1))
            << "idef with a perturbed bond has a perturbed bond";
    EXPECT_EQ(1, idef.numNonperturbedInteractions[F_LJ14] / (NRAL(F_LJ14) + 1))
            << "idef with all perturbed LJ 1-4 has no non-perturbed LJ 1-4";
    EXPECT_THAT(idef.il[F_BONDS].iatoms, Pointwise(Eq(), { 1, 0, 1, 0, 2, 3 }));
    EXPECT_THAT(idef.il[F_LJ14].iatoms, Pointwise(Eq(), { 4, 0, 1, 2, 2, 3, 3, 2, 3 }));
}

TEST(TopSortTest, SortsMoreComplexIdefWithPerturbedInteractions)
{
    // Interleave non-perturbed and perturbed interactions, and
    // observe that the perturbed interactions get sorted to the end
    // of the interaction lists.
    gmx_ffparams_t         forcefieldParams;
    InteractionDefinitions idef(forcefieldParams);

    // F_BONDS
    std::array<int, 2> bondAtoms = { 0, 1 };
    forcefieldParams.iparams.push_back(makeUnperturbedBondParams(0.9, 1000));
    idef.il[F_BONDS].push_back(forcefieldParams.iparams.size() - 1, bondAtoms);
    std::array<int, 2> perturbedBondAtoms = { 2, 3 };
    forcefieldParams.iparams.push_back(makePerturbedBondParams(0.9, 1000, 1.1, 4000));
    idef.il[F_BONDS].push_back(forcefieldParams.iparams.size() - 1, perturbedBondAtoms);
    std::array<int, 2> moreBondAtoms = { 4, 5 };
    forcefieldParams.iparams.push_back(makeUnperturbedBondParams(0.9, 1000));
    idef.il[F_BONDS].push_back(forcefieldParams.iparams.size() - 1, moreBondAtoms);
    std::array<int, 2> morePerturbedBondAtoms = { 6, 7 };
    forcefieldParams.iparams.push_back(makePerturbedBondParams(0.8, 100, 1.2, 6000));
    idef.il[F_BONDS].push_back(forcefieldParams.iparams.size() - 1, morePerturbedBondAtoms);

    // F_LJ14
    forcefieldParams.iparams.push_back(makeUnperturbedLJ14Params(100, 10000));
    idef.il[F_LJ14].push_back(forcefieldParams.iparams.size() - 1, perturbedBondAtoms);
    forcefieldParams.iparams.push_back(makeUnperturbedLJ14Params(100, 10000));
    idef.il[F_LJ14].push_back(forcefieldParams.iparams.size() - 1, bondAtoms);
    forcefieldParams.iparams.push_back(makePerturbedLJ14Params(100, 10000, 200, 20000));
    idef.il[F_LJ14].push_back(forcefieldParams.iparams.size() - 1, morePerturbedBondAtoms);
    forcefieldParams.iparams.push_back(makeUnperturbedLJ14Params(100, 10000));
    idef.il[F_LJ14].push_back(forcefieldParams.iparams.size() - 1, moreBondAtoms);
    // Perturb the charge of atom 2, affecting the non-perturbed LJ14 above
    std::vector<int64_t> atomInfo{ 0, 0, sc_atomInfo_HasPerturbedCharge, 0, 0, 0, 0, 0 };

    gmx_sort_ilist_fe(&idef, atomInfo);

    EXPECT_EQ(2, idef.numNonperturbedInteractions[F_BONDS] / (NRAL(F_BONDS) + 1))
            << "idef with some perturbed bonds has some perturbed bonds";
    EXPECT_EQ(2, idef.numNonperturbedInteractions[F_LJ14] / (NRAL(F_LJ14) + 1))
            << "idef with some perturbed LJ 1-4 has some non-perturbed LJ 1-4";
    EXPECT_THAT(idef.il[F_BONDS].iatoms, Pointwise(Eq(), { 0, 0, 1, 2, 4, 5, 1, 2, 3, 3, 6, 7 }));
    EXPECT_THAT(idef.il[F_LJ14].iatoms, Pointwise(Eq(), { 5, 0, 1, 7, 4, 5, 4, 2, 3, 6, 6, 7 }));
}

} // namespace

} // namespace gmx
