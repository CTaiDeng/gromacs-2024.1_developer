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

/*!\file
 * \brief
 * This file declares functions for setting up a benchmark system
 *
 * \author Berk Hess <hess@kth.se>
 * \ingroup module_nbnxm
 */

#ifndef GMX_NBNXN_BENCH_SYSTEM_H
#define GMX_NBNXN_BENCH_SYSTEM_H

#include <string>
#include <vector>

#include "gromacs/math/vectypes.h"
#include "gromacs/mdtypes/forcerec.h"
#include "gromacs/utility/listoflists.h"
#include "gromacs/utility/smalloc.h"

namespace gmx
{

//! Description of the system used for benchmarking.
struct BenchmarkSystem
{
    /*! \brief Constructor
     *
     * Generates a benchmark system of size \p multiplicationFactor
     * times the base size by stacking cubic boxes of 1000 water molecules
     * with 3000 atoms total.
     *
     * \param[in] multiplicationFactor  Should be a power of 2, is checked
     * \param[in] outputFile            The name of the csv file to write benchmark results
     */
    BenchmarkSystem(int multiplicationFactor, const std::string& outputFile);

    //! Number of different atom types in test system.
    int numAtomTypes;
    //! Storage for parameters for short range interactions.
    std::vector<real> nonbondedParameters;
    //! Storage for atom type parameters.
    std::vector<int> atomTypes;
    //! Storage for atom partial charges.
    std::vector<real> charges;
    //! Atom info where all atoms are marked to have Van der Waals interactions
    std::vector<int64_t> atomInfoAllVdw;
    //! Atom info where only oxygen atoms are marked to have Van der Waals interactions
    std::vector<int64_t> atomInfoOxygenVdw;
    //! Information about exclusions.
    ListOfLists<int> excls;
    //! Storage for atom positions.
    std::vector<gmx::RVec> coordinates;
    //! System simulation box.
    matrix box;
    //! Forcerec with only the entries used in the benchmark set
    t_forcerec forceRec;
    //! csv output file
    FILE* csv;
};

} // namespace gmx

#endif
