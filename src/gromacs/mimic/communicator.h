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

#ifndef GMX_MIMIC_COMMUNICATOR_H
#define GMX_MIMIC_COMMUNICATOR_H

#include "gromacs/mdlib/constr.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/futil.h"

namespace gmx
{

template<class T>
class ArrayRef;

/**
 * \inlibraryapi
 * \internal \brief
 * Class-wrapper around MiMiC communication library
 * It uses GROMACS' unit conversion to switch from GROMACS' units to a.u.
 *
 * \author Viacheslav Bolnykh <v.bolnykh@hpc-leap.eu>
 * \ingroup module_mimic
 */
class MimicCommunicator
{

public:
    /*! \brief
     * Initializes the communicator
     */
    static void init();

    /*! \brief
     * Sends the data needed for MiMiC initialization
     *
     * That includes number of atoms, element numbers, charges, masses,
     * maximal order of multipoles (0 for point-charge forcefields),
     * number of molecules, number of atoms per each molecule,
     * bond constraints data
     *
     * @param mtop global topology data
     * @param coords coordinates of all atoms
     */
    static void sendInitData(gmx_mtop_t* mtop, ArrayRef<const RVec> coords);

    /*! \brief
     * Gets the number of MD steps to perform from MiMiC
     *
     * @return nsteps the number of MD steps to perform
     */
    static int64_t getStepNumber();

    /*! \brief
     * Receive and array of updated atomic coordinates from MiMiC
     *
     * @param x array of coordinates to fill
     * @param natoms number of atoms in the system
     */
    static void getCoords(ArrayRef<RVec> x, int natoms);

    /*! \brief
     * Send the potential energy value to MiMiC
     *
     * @param energy energy value to send
     */
    static void sendEnergies(real energy);

    /*! \brief
     * Send classical forces acting on all atoms in the system
     * to MiMiC.
     *
     * @param forces array of forces to send
     * @param natoms number of atoms in the system
     */
    static void sendForces(ArrayRef<gmx::RVec> forces, int natoms);

    /*! \brief
     * Finish communications and disconnect from the server
     */
    static void finalize();
};

} // namespace gmx

#endif // GMX_MIMIC_COMMUNICATOR_H
