/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2014- The GROMACS Authors
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

#ifndef GMX_SIMD_TESTS_DATA_H
#define GMX_SIMD_TESTS_DATA_H

/*! \internal \file
 * \brief Common test data constants for SIMD, SIMD4 and scalar tests
 *
 * To avoid silent bugs due to double/float truncation if we ever use the
 * wrong return type of routines, we want to use test data that fills all
 * available bits in either single or double precision. The values themselves
 * are meaningless.
 * Note that the data is used to initialize the SIMD constants, which for
 * technical (alignment) reasons in some compilers cannot be placed inside
 * the text fixture classes. For that reason this data cannot go in the
 * fixtures either.
 *
 * \author Erik Lindahl <erik.lindahl@scilifelab.se>
 * \ingroup module_simd
 */

#include "gromacs/utility/real.h"

namespace gmx
{
namespace test
{

/*! \cond internal */
/*! \addtogroup module_simd */
/*! \{ */
constexpr real czero = 0.0;                //!< zero
constexpr real c0    = 0.3333333333333333; //!< test constant 0.0 + 1.0/3.0
constexpr real c1    = 1.7142857142857144; //!< test constant 1.0 + 5.0/7.0
constexpr real c2    = 2.6923076923076925; //!< test constant 2.0 + 9.0/13.0
constexpr real c3    = 3.8947368421052633; //!< test constant 3.0 + 17.0/19.0
constexpr real c4    = 4.793103448275862;  //!< test constant 4.0 + 23.0/29.0
constexpr real c5    = 5.837837837837838;  //!< test constant 5.0 + 31.0/37.0
constexpr real c6    = 6.953488372093023;  //!< test constant 6.0 + 41.0/43.0
constexpr real c7    = 7.886792452830189;  //!< test constant 7.0 + 47.0/53.0
constexpr real c8    = 8.967213114754099;  //!< test constant 8.0 + 59.0/61.0

/*! \} */
/*! \endcond */

} // namespace test
} // namespace gmx

#endif
