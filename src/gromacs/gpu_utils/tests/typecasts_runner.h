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
 * Header for runner for CUDA float3 type layout tests.
 *
 * \author Artem Zhmurov <zhmurov@gmail.com>
 */
#ifndef GMX_GPU_UTILS_TESTS_TYPECASTS_RUNNER_H
#define GMX_GPU_UTILS_TESTS_TYPECASTS_RUNNER_H

#include "gmxpre.h"

#include <vector>

#include <gtest/gtest.h>

#include "gromacs/math/vectypes.h"

#include "testutils/test_device.h"

namespace gmx
{

template<typename>
class ArrayRef;

namespace test
{

/*! \brief Tests the compatibility of RVec and float3 using the conversion on host.
 *
 * \param[out] rVecOutput  Data in RVec format for the output.
 * \param[in]  rVecInput   Data in RVec format with the input.
 */
void convertRVecToFloat3OnHost(ArrayRef<gmx::RVec> rVecOutput, ArrayRef<const gmx::RVec> rVecInput);

/*! \brief Tests the compatibility of RVec and float3 using the conversion on device.
 *
 * \param[out] rVecOutput  Data in RVec format for the output.
 * \param[in]  rVecInput   Data in RVec format with the input.
 * \param[in]  testDevice  Test herdware environment to get DeviceContext and DeviceStream from.
 */
void convertRVecToFloat3OnDevice(ArrayRef<gmx::RVec>       rVecOutput,
                                 ArrayRef<const gmx::RVec> rVecInput,
                                 const TestDevice*         testDevice);


} // namespace test
} // namespace gmx

#endif // GMX_GPU_UTILS_TESTS_TYPECASTS_RUNNER_H
