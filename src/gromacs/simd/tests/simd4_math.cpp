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
#include <cstdint>

#include <vector>

#include "gromacs/math/units.h"
#include "gromacs/math/utilities.h"
#include "gromacs/options/basicoptions.h"
#include "gromacs/simd/simd.h"
#include "gromacs/simd/simd_math.h"

#include "simd4.h"

#if GMX_SIMD

namespace gmx
{
namespace test
{


#    if GMX_SIMD4_HAVE_REAL

/*! \cond internal */
/*! \addtogroup module_simd */
/*! \{ */

class Simd4MathTest : public Simd4Test
{
};

/*! \} */
/*! \endcond */

// Actual math function tests below

namespace
{

/*! \cond internal */
/*! \addtogroup module_simd */
/*! \{ */

// Presently, the only SIMD4 math function is 1/sqrt(x), which
// has a close-to-trivial implementation without different
// approximation intervals or special threshold points. To
// avoid having to re-implement the entire SIMD math function
// test infrastructure we only test these functions for a few
// values that are either special or exercise all bits.

TEST_F(Simd4MathTest, invsqrt)
{
    const real x0 = std::numeric_limits<float>::min();
    const real x1 = std::numeric_limits<float>::max();
    const real x2 = M_PI;

    GMX_EXPECT_SIMD4_REAL_NEAR(setSimd4RealFrom3R(1.0 / sqrt(x0), 1.0 / sqrt(x1), 1.0 / sqrt(x2)),
                               invsqrt(setSimd4RealFrom3R(x0, x1, x2)));
}

TEST_F(Simd4MathTest, invsqrtSingleAccuracy)
{
    const real x0 = std::numeric_limits<float>::min();
    const real x1 = std::numeric_limits<float>::max();
    const real x2 = M_PI;

    /* Increase the allowed error by the difference between the actual precision and single */
    setUlpTolSingleAccuracy(ulpTol_);

    GMX_EXPECT_SIMD4_REAL_NEAR(setSimd4RealFrom3R(1.0 / sqrt(x0), 1.0 / sqrt(x1), 1.0 / sqrt(x2)),
                               invsqrtSingleAccuracy(setSimd4RealFrom3R(x0, x1, x2)));
}

/*! \} */
/*! \endcond */

} // namespace

#    endif // GMX_SIMD4_HAVE_REAL

} // namespace test
} // namespace gmx

#endif // GMX_SIMD
