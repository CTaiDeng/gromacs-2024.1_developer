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
 * Declares the Colvars GROMACS proxy class
 *
 * \author Hubert Santuz <hubert.santuz@gmail.com>
 * \ingroup module_applied_forces
 */

#ifndef GMX_APPLIED_FORCES_COLVARPROXYGROMACS_H
#define GMX_APPLIED_FORCES_COLVARPROXYGROMACS_H

// NOLINTBEGIN
// Disabling clang-tidy checks on Colvars library code that is not called directly by GROMACS,
// or is not even used at all (e.g. code used by NAMD or VMD interfaces)
#include "external/colvars/colvarproxy.h"
// NOLINTEND

#include "gromacs/pbcutil/pbc.h"
#include "gromacs/random/tabulatednormaldistribution.h"
#include "gromacs/random/threefry.h"
#include "gromacs/topology/atoms.h"
#include "gromacs/utility/logger.h"


namespace gmx
{


/*! \internal \brief
 * Implements a GROMACS version of colvarproxy.
 * This class hold for the communication between colvars and GROMACS.
 * 2 child class will inherit from this one: one during pre processing (ColvarsPreProcessor)
 * and one during the simulation (ColvarsForceProvider).
 * Most of the work needed for the communication will be implemented in this class.
 */
class ColvarProxyGromacs : public colvarproxy
{

protected:
    //! Atoms topology
    t_atoms gmxAtoms_;

    //! Box infos
    PbcType pbcType_;
    t_pbc   gmxPbc_;

    // GROMACS logger instance
    const MDLogger* logger_ = nullptr;

    //! Activate or not the parsing of the Colvars config file
    bool doParsing_;


    // GROMACS random number generation.
    DefaultRandomEngine           rng_; // gromacs random number generator
    TabulatedNormalDistribution<> normalDistribution_;


public:
    friend class cvm::atom;

    /*! \brief Construct ColvarProxyGromacs from its parameters
     *
     * \param[in] colvarsConfigString Content of the colvars input file.
     * \param[in] atoms Atoms topology
     * \param[in] pbcType Periodic boundary conditions
     * \param[in] logger GROMACS logger instance
     * \param[in] doParsing Wether the input file should be parsed.
     * \param[in] inputStrings Input files stored as string in the TPR's KVT
     * \param[in] ensembleTemperature the constant ensemble temperature
     * \param[in] seed the colvars seed for random number generator
     */
    ColvarProxyGromacs(const std::string&                        colvarsConfigString,
                       t_atoms                                   atoms,
                       PbcType                                   pbcType,
                       const MDLogger*                           logger,
                       bool                                      doParsing,
                       const std::map<std::string, std::string>& inputStrings,
                       real                                      ensembleTemperature,
                       int                                       seed);
    ~ColvarProxyGromacs() override;

    //! Update colvars topology of one atom mass and charge from the GROMACS topology
    void updateAtomProperties(int index);

    //! From colvarproxy
    // Methods below override virtual ones present in the `colvarproxy` class

    //! Return a random number from a Gaussian distribution
    cvm::real rand_gaussian() override;

    //! Print a message to the main log
    void log(std::string const& message) override;
    //! Print a message to the main log and let GROMACS handle the error
    void error(std::string const& message) override;
    //! Request to set the units used internally by Colvars
    int set_unit_system(std::string const& unitsIn, bool colvarsDefined) override;

    //! Initialize colvars atom from GROMACS topology
    int init_atom(int atomNumber) override;
    /*! \brief Check if atom belongs to the global index of atoms
     *  \param[in] atomNumber Colvars index of the atom to check
     */
    int check_atom_id(int atomNumber) override;

    //! Compute the minimum distance with respect to the PBC between 2 atoms.
    cvm::rvector position_distance(cvm::atom_pos const& pos1, cvm::atom_pos const& pos2) const override;
};

} // namespace gmx

#endif // GMX_APPLIED_FORCES_COLVARPROXYGROMACS_H
