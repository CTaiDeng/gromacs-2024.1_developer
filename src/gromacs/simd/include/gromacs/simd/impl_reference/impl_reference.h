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

#ifndef GMX_SIMD_IMPL_REFERENCE_H
#define GMX_SIMD_IMPL_REFERENCE_H

/*! \libinternal \file
 *
 * \brief Reference SIMD implementation, including SIMD documentation.
 *
 * \author Erik Lindahl <erik.lindahl@scilifelab.se>
 *
 * \ingroup module_simd
 */


// Definitions for this architecture
#include "impl_reference_definitions.h"

// Functions not related to floating-point SIMD (mainly prefetching)
#include "impl_reference_general.h"

// Special width-4 double-precision SIMD
#include "impl_reference_simd4_double.h"

// Special width-4 single-precision SIMD
#include "impl_reference_simd4_float.h"

// General double-precision SIMD (and double/float conversions)
#include "impl_reference_simd_double.h"

// General single-precision SIMD
#include "impl_reference_simd_float.h"

// Higher-level utility functions for double precision SIMD
#include "impl_reference_util_double.h"

// Higher-level utility functions for single precision SIMD
#include "impl_reference_util_float.h"

#endif // GMX_SIMD_IMPL_REFERENCE_H
