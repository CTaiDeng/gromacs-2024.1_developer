/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2015- The GROMACS Authors
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

/*! \file
 * \brief Random seed and domain utilities
 *
 * This file contains utilities to create true random seeds from the system,
 * and logic to keep track of different random domains for random engines such
 * as ThreeFry that can take a second seed value.
 *
 * \author Erik Lindahl <erik.lindahl@gmail.com>
 * \inpublicapi
 * \ingroup module_random
 */

#ifndef GMX_RANDOM_SEED_H
#define GMX_RANDOM_SEED_H

#include <random>

#include "gromacs/utility/basedefinitions.h"

namespace gmx
{

/*! \brief Return 64 random bits from the random device, suitable as seed.
 *
 *  If the internal random device output is smaller than 64 bits, this routine
 *  will use multiple calls internally until we have 64 bits of random data.
 *
 *  \return 64-bit unsigned integer with random bits.
 */
uint64_t makeRandomSeed();

/*! \brief Random device
 *
 *  For now this is identical to the standard library, but since we use
 *  the GROMACS random module for all other random engines and distributions
 *  it is convenient to have this too in the same module.
 */
typedef std::random_device RandomDevice;

/*! \brief Enumerated values for fixed part of random seed (domain)
 *
 *  Random numbers are used in many places in GROMACS, and to avoid identical
 *  streams the random seeds should be different. Instead of keeping track of
 *  several different user-provided seeds, it is better to use the fact that
 *  generators like ThreeFry take two 64-bit keys, and combine a general
 *  user-provided 64-bit random seed with a second constant value from this list
 *  to make each stream guaranteed unique.
 *
 *  \note There is no reason to go overboard with adding options; we only
 *        need to guarantee different streams for cases that might be present
 *        simultaneously in a single simulation. As an example, two different
 *        integrators (or thermostats) can reuse the same domain.
 *  \note When you do add options, leave some space between the values so
 *        you can group new options with old ones without changing old values.
 */
enum class RandomDomain
{
    Other                 = 0x00000000, //!< Generic - stream uniqueness is not important
    MaxwellVelocities     = 0x00001000, //!< Veolcity assignment from Maxwell distribution
    TestParticleInsertion = 0x00002000, //!< Test particle insertion
    UpdateCoordinates     = 0x00003000, //!< Particle integrators
    UpdateConstraints     = 0x00004000, //!< Second integrator step for constraints
    Thermostat            = 0x00005000, //!< Stochastic temperature coupling
    Barostat              = 0x00006000, //!< Stochastic pressure coupling
    ReplicaExchange       = 0x00007000, //!< Replica exchange metropolis moves
    ExpandedEnsemble      = 0x00008000, //!< Expanded ensemble lambda moves
    AwhBiasing            = 0x00009000  //!< AWH biasing reference value moves
};

} // namespace gmx

#endif // GMX_RANDOM_SEED_H
