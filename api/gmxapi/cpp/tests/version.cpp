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

#include "gmxapi/version.h"

#include <climits>

#include "testingconfiguration.h"

namespace gmxapi
{

namespace testing
{

namespace
{

using gmxapi::Version;

/* Copy header version info with intentionally sloppy type-ing to try to catch
 * worst-case scenarios and unexpected behavior. Otherwise the isAtLeast function
 * uses major(), minor(), and patch() so testing them might be superfluous.
 */
//! \cond
const int current_major = gmxapi::c_majorVersion;
const int current_minor = gmxapi::c_minorVersion;
const int current_patch = gmxapi::c_patchVersion;
//! \endcond

/*!
 * \brief Check basic Version interface functionality.
 */
TEST_F(GmxApiTest, SaneVersionComparisons)
{
    EXPECT_TRUE(Version::isAtLeast(0, 0, 0));
    EXPECT_FALSE(Version::isAtLeast(SHRT_MAX, SHRT_MAX, SHRT_MAX));
    EXPECT_TRUE(Version::isAtLeast(current_major, current_minor, current_patch));
    EXPECT_FALSE(Version::isAtLeast(current_major + 1, current_minor, current_patch));
    EXPECT_FALSE(Version::isAtLeast(current_major, current_minor + 1, current_patch));
    EXPECT_FALSE(Version::isAtLeast(current_major, current_minor, current_patch + 1));
}

/*!
 * \brief Check whether gmxapi correctly advertises or refutes feature availability.
 *
 * Check for correct responses from the Version API for features or
 * functionality not (yet) guaranteed by the current API version.
 * If a feature is available, it is expected to conform to the API specification
 * for the library Version::release(). As we discover features that break
 * forward-compatibility of the API, we will have to provide developer documentation
 * or sample code for build-time CMake feature checks.
 *
 * This is the test for pre-0.1 features leading up to that specification.
 *
 * \internal
 * Designed but unimplemented features should be tested with ``EXPECT_FALSE``
 * until they are implemented, then toggled to ``EXPECT_TRUE`` as implemented as
 * extensions of the current API spec. (There aren't any yet.)
 */
TEST_F(GmxApiTest, VersionNamed0_1_Features)
{
    EXPECT_FALSE(Version::hasFeature(""));
    EXPECT_FALSE(Version::hasFeature("nonexistent feature"));
}

} // end anonymous namespace

} // namespace testing

} // namespace gmxapi
