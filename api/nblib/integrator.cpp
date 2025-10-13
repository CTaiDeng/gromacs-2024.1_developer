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

/*! \internal \file
 * \brief
 * Implements nblib integrator
 *
 * \author Victor Holanda <victor.holanda@cscs.ch>
 * \author Joe Jordan <ejjordan@kth.se>
 * \author Prashanth Kanduri <kanduri@cscs.ch>
 * \author Sebastian Keller <keller@cscs.ch>
 * \author Artem Zhmurov <zhmurov@gmail.com>
 */
#include "nblib/integrator.h"

#include "gromacs/pbcutil/pbc.h"
#include "gromacs/utility/arrayref.h"

#include "nblib/topology.h"

namespace nblib
{

LeapFrog::LeapFrog(const Topology& topology, const Box& box) : box_(box)
{
    inverseMasses_.resize(topology.numParticles());
    for (int i = 0; i < topology.numParticles(); i++)
    {
        int typeIndex     = topology.getParticleTypeIdOfAllParticles()[i];
        inverseMasses_[i] = 1.0 / topology.getParticleTypes()[typeIndex].mass();
    }
}

LeapFrog::LeapFrog(gmx::ArrayRef<const real> inverseMasses, const Box& box) :
    inverseMasses_(inverseMasses.begin(), inverseMasses.end()), box_(box)
{
}

void LeapFrog::integrate(const real dt, gmx::ArrayRef<Vec3> x, gmx::ArrayRef<Vec3> v, gmx::ArrayRef<const Vec3> f)
{
    for (size_t i = 0; i < x.size(); i++)
    {
        for (int dim = 0; dim < dimSize; dim++)
        {
            v[i][dim] += f[i][dim] * dt * inverseMasses_[i];
            x[i][dim] += v[i][dim] * dt;
        }
    }
}

} // namespace nblib
