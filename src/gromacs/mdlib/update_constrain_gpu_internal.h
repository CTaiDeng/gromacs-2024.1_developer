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

/*! \internal \file
 *
 * \brief Declares GPU implementations of backend-specific update-constraints functions.
 *
 * \author Artem Zhmurov <zhmurov@gmail.com>
 *
 * \ingroup module_mdlib
 */
#ifndef GMX_MDLIB_UPDATE_CONSTRAIN_GPU_INTERNAL_H
#define GMX_MDLIB_UPDATE_CONSTRAIN_GPU_INTERNAL_H

#include "gmxpre.h"

#include "gromacs/gpu_utils/device_stream.h"
#include "gromacs/gpu_utils/devicebuffer_datatype.h"
#include "gromacs/gpu_utils/gputraits.h"
#include "gromacs/math/matrix.h"

class GpuEventSynchronizer;

namespace gmx
{

/*! \internal \brief Scaling matrix struct.
 *
 * \todo Should be generalized.
 */
struct ScalingMatrix
{
    ScalingMatrix(const Matrix3x3& m) :
        xx(m(XX, XX)), yy(m(YY, YY)), zz(m(ZZ, ZZ)), yx(m(YY, XX)), zx(m(ZZ, XX)), zy(m(ZZ, YY))
    {
    }
    float xx, yy, zz, yx, zx, zy;
};

/*! \brief Launches positions of velocities scaling kernel.
 *
 * \param[in] numAtoms       Number of atoms in the system.
 * \param[in] d_coordinates  Device buffer with position or velocities to be scaled.
 * \param[in] mu             Scaling matrix.
 * \param[in] deviceStream   Stream to launch kernel in.
 */
void launchScaleCoordinatesKernel(int                  numAtoms,
                                  DeviceBuffer<Float3> d_coordinates,
                                  const ScalingMatrix& mu,
                                  const DeviceStream&  deviceStream);

} // namespace gmx

#endif // GMX_MDLIB_UPDATE_CONSTRAIN_GPU_INTERNAL_H
