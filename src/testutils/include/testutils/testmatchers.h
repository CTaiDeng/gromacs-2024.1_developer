/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2018- The GROMACS Authors
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

/*! \libinternal \file
 * \brief Extra GoogleMock matchers for unit tests.
 *
 * This file provides the usual kind of GoogleMock matchers that
 * extend the usefulness of GoogleMock EXPECT_THAT constructs to the
 * kinds of containers of reals commonly used. This means that test
 * code can write one-liners rather than loops over whole containers.
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \inlibraryapi
 * \ingroup module_testutils
 */
#ifndef GMX_TESTUTILS_TESTMATCHERS_H
#define GMX_TESTUTILS_TESTMATCHERS_H

#include <memory>
#include <ostream>
#include <tuple>

#include <gmock/gmock.h>

#include "gromacs/utility/real.h"

namespace gmx
{
template<typename T>
class BasicVector;

namespace test
{

class FloatingPointTolerance;

/*! \brief Make matcher for floats for use with GoogleMock that compare
 * equal when \c tolerance is satisifed.
 *
 * Used like
 *
 *   EXPECT_THAT(testFloats, Pointwise(FloatEq(tolerance), referenceFloats));
 */
testing::Matcher<std::tuple<float, float>> FloatEq(const FloatingPointTolerance& tolerance);

/*! \brief Make matcher for doubles for use with GoogleMock that compare
 * equal when \c tolerance is satisifed.
 *
 * Used like
 *
 *   EXPECT_THAT(testDoubles, Pointwise(DoubleEq(tolerance), referenceDoubles));
 */
testing::Matcher<std::tuple<double, double>> DoubleEq(const FloatingPointTolerance& tolerance);

/*! \brief Make matcher for reals for use with GoogleMock that compare
 * equal when \c tolerance is satisifed.
 *
 * Used like
 *
 *   EXPECT_THAT(testReals, Pointwise(RealEq(tolerance), referenceReals));
 */
testing::Matcher<std::tuple<real, real>> RealEq(const FloatingPointTolerance& tolerance);

/*! \brief Make matcher for RVecs for use with GoogleMock that compare
 * equal when \c tolerance is satisifed.
 *
 * Used like
 *
 *   EXPECT_THAT(testRVecs, Pointwise(RVecEq(tolerance), referenceRVecs));
 */
testing::Matcher<std::tuple<BasicVector<real>, BasicVector<real>>> RVecEq(const FloatingPointTolerance& tolerance);

} // namespace test
} // namespace gmx

#endif
