/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2020- The GROMACS Authors
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

/*! \inpublicapi \file
 * \brief
 * Implements ParticleSequencer class
 *
 * \author Victor Holanda <victor.holanda@cscs.ch>
 * \author Joe Jordan <ejjordan@kth.se>
 * \author Prashanth Kanduri <kanduri@cscs.ch>
 * \author Sebastian Keller <keller@cscs.ch>
 * \author Artem Zhmurov <zhmurov@gmail.com>
 */
#ifndef NBLIB_PARTICLESEQUENCER_H
#define NBLIB_PARTICLESEQUENCER_H

#include <tuple>
#include <unordered_map>
#include <vector>

#include "nblib/molecules.h"

namespace nblib
{

//! Helper class for Topology to keep track of particle IDs
class ParticleSequencer
{
    //! Alias for storing by (molecule name, molecule nr, residue name, particle name)
    using DataType = std::unordered_map<
            std::string,
            std::unordered_map<int, std::unordered_map<std::string, std::unordered_map<std::string, int>>>>;

public:
    //! Build sequence from a list of molecules
    void build(const std::vector<std::tuple<Molecule, int>>& moleculesList);

    //! Access ID by (molecule name, molecule nr, residue name, particle name)
    int operator()(const MoleculeName&, int, const ResidueName&, const ParticleName&) const;

private:
    DataType data_;
};

} // namespace nblib

#endif // NBLIB_PARTICLESEQUENCER_H
