/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2014- The GROMACS Authors
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

#include "gmxpre.h"

#include <cmath>

#include "gromacs/simd/simd.h"
#include "gromacs/simd/vector_operations.h"

#include "data.h"
#include "simd4.h"

#if GMX_SIMD

namespace gmx
{
namespace test
{
namespace
{

/*! \cond internal */
/*! \addtogroup module_simd */
/*! \{ */

#    if GMX_SIMD4_HAVE_REAL

/*! \brief Test fixture for SIMD4 vector operations (identical to the SIMD4 \ref Simd4Test) */
typedef Simd4Test Simd4VectorOperationsTest;

TEST_F(Simd4VectorOperationsTest, norm2)
{
    Simd4Real simdX  = rSimd4_c0c1c2;
    Simd4Real simdY  = rSimd4_c3c4c5;
    Simd4Real simdZ  = rSimd4_c6c7c8;
    Simd4Real simdR2 = setSimd4RealFrom3R(
            c0 * c0 + c3 * c3 + c6 * c6, c1 * c1 + c4 * c4 + c7 * c7, c2 * c2 + c5 * c5 + c8 * c8);

    setUlpTol(2);
    GMX_EXPECT_SIMD4_REAL_NEAR(simdR2, norm2(simdX, simdY, simdZ));
}

#    endif // GMX_SIMD4_HAVE_REAL

/*! \} */
/*! \endcond */

} // namespace
} // namespace test
} // namespace gmx

#endif // GMX_SIMD
