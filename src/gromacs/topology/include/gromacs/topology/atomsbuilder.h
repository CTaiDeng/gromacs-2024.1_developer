/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2014- The GROMACS Authors
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
 * \brief
 * Utility classes for manipulating \c t_atoms structures.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \inlibraryapi
 */
#ifndef GMX_TOPOLOGY_ATOMSBUILDER_H
#define GMX_TOPOLOGY_ATOMSBUILDER_H

#include <memory>
#include <vector>

#include "gromacs/math/vectypes.h"
#include "gromacs/utility/classhelpers.h"
#include "gromacs/utility/real.h"

struct t_atoms;
struct t_resinfo;
struct t_symtab;

namespace gmx
{

class AtomsBuilder
{
public:
    AtomsBuilder(t_atoms* atoms, t_symtab* symtab);
    ~AtomsBuilder();

    void reserve(int atomCount, int residueCount);
    void clearAtoms();

    int currentAtomCount() const;

    void setNextResidueNumber(int number);
    void addAtom(const t_atoms& atoms, int i);
    void startResidue(const t_resinfo& resinfo);
    void finishResidue(const t_resinfo& resinfo);
    void discardCurrentResidue();

    void mergeAtoms(const t_atoms& atoms);

private:
    char** symtabString(char** source);

    t_atoms*  atoms_;
    t_symtab* symtab_;
    int       nrAlloc_;
    int       nresAlloc_;
    int       currentResidueIndex_;
    int       nextResidueNumber_;

    GMX_DISALLOW_COPY_AND_ASSIGN(AtomsBuilder);
};

class AtomsRemover
{
public:
    explicit AtomsRemover(const t_atoms& atoms);
    ~AtomsRemover();

    void refreshAtomCount(const t_atoms& atoms);

    void markAll();
    void markResidue(const t_atoms& atoms, int atomIndex, bool bStatus);
    bool isMarked(int atomIndex) const { return removed_[atomIndex] != 0; }

    void removeMarkedElements(std::vector<RVec>* container) const;
    void removeMarkedElements(std::vector<real>* container) const;
    void removeMarkedAtoms(t_atoms* atoms) const;

private:
    std::vector<char> removed_;

    GMX_DISALLOW_COPY_AND_ASSIGN(AtomsRemover);
};

} // namespace gmx

#endif
