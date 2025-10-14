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
 * Declares the Colvars GROMACS proxy class during pre-processing.
 *
 * \author Hubert Santuz <hubert.santuz@gmail.com>
 * \ingroup module_applied_forces
 */
#ifndef GMX_APPLIED_FORCES_COLVARSPREPROCESSOR_H
#define GMX_APPLIED_FORCES_COLVARSPREPROCESSOR_H


#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/keyvaluetreebuilder.h"

#include "colvarproxygromacs.h"


namespace gmx
{

/*! \internal \brief
 * Class that read a colvars configuration file during pre-processing and
 * retrieve the colvars atoms coordinates to be stored in tpr KVT.
 */
class ColvarsPreProcessor : public ColvarProxyGromacs
{
public:
    /*! \brief Construct ColvarsPreProcessor from its parameters
     *

     * \param[in] colvarsConfigString Content of the colvars input file.
     * \param[in] atoms Atoms topology
     * \param[in] pbcType Periodic boundary conditions
     * \param[in] logger GROMACS logger instance
     * \param[in] ensembleTemperature the constant ensemble temperature
     * \param[in] seed the colvars seed for random number generator
     * \param[in] box Matrix with full box of the system
     * \param[in] x Coordinates of each atom in the system
     */
    ColvarsPreProcessor(const std::string&   colvarsConfigString,
                        t_atoms              atoms,
                        PbcType              pbcType,
                        const MDLogger*      logger,
                        real                 ensembleTemperature,
                        int                  seed,
                        const matrix         box,
                        ArrayRef<const RVec> x);


    //! Return a vector of the colvars atoms coordinates
    std::vector<RVec> getColvarsCoords();

    //! Save all input files of colvars (outside the config file) in the tpr file through the key-value-tree
    bool inputStreamsToKVT(KeyValueTreeObjectBuilder treeBuilder, const std::string& tag);

private:
    //! Atoms coordinates of the whole system
    ArrayRef<const RVec> x_;
};


} // namespace gmx

#endif // GMX_APPLIED_FORCES_COLVARSPREPROCESSOR_H
