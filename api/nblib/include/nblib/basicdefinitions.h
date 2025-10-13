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

/*! \inpublicapi \file
 * \brief
 * Implements some definitions that are identical to those of gromacs
 *
 * \author Victor Holanda <victor.holanda@cscs.ch>
 * \author Joe Jordan <ejjordan@kth.se>
 * \author Prashanth Kanduri <kanduri@cscs.ch>
 * \author Sebastian Keller <keller@cscs.ch>
 * \author Artem Zhmurov <zhmurov@gmail.com>
 */
#ifndef NBLIB_BASICDEFINITIONS_H
#define NBLIB_BASICDEFINITIONS_H

#include <cmath>

// from utility/real.h
#if GMX_DOUBLE
#    ifndef HAVE_REAL
typedef double real;
#        define HAVE_REAL
#    endif
#else /* GMX_DOUBLE */
#    ifndef HAVE_REAL
typedef float real;
#        define HAVE_REAL
#    endif
#endif /* GMX_DOUBLE */

namespace nblib
{

namespace detail
{
// from math/units.h
constexpr const float  KILO      = 1e3;             /* Thousand	*/
constexpr const double NANO      = 1e-9;            /* A Number	*/
constexpr const double E_CHARGE  = 1.602176634e-19; /* Exact definition, Coulomb NIST 2018 CODATA */
constexpr const double BOLTZMANN = 1.380649e-23;    /* (J/K, Exact definition, NIST 2018 CODATA */
constexpr const double AVOGADRO  = 6.02214076e23;   /* 1/mol, Exact definition, NIST 2018 CODATA */
constexpr const double RGAS      = (BOLTZMANN * AVOGADRO); /* (J/(mol K))  */
constexpr const double EPSILON0_SI = 8.8541878128e-12;     /* F/m,  NIST 2018 CODATA */
constexpr const double EPSILON0 = ((EPSILON0_SI * NANO * KILO) / (E_CHARGE * E_CHARGE * AVOGADRO));

// from pbc/ishift.h
constexpr const int D_BOX_Z = 1;
constexpr const int D_BOX_Y = 1;
constexpr const int D_BOX_X = 2;
constexpr const int N_BOX_Z = (2 * D_BOX_Z + 1);
constexpr const int N_BOX_Y = (2 * D_BOX_Y + 1);
constexpr const int N_BOX_X = (2 * D_BOX_X + 1);
constexpr const int N_IVEC  = (N_BOX_Z * N_BOX_Y * N_BOX_X);
} // namespace detail

//! Needed for generating Bolzmann velocity distribution (kJ/(mol K))
constexpr const real BOLTZ = (detail::RGAS / detail::KILO); /*  */

//! Charge multiplication factor for Coulomb interactions
constexpr const real ONE_4PI_EPS0 = (1.0 / (4.0 * M_PI * detail::EPSILON0));

//! Conversion factor from degrees to radians
constexpr const real DEG2RAD = M_PI / 180.0;

//! The number of shift vectors needed for pbc
constexpr const int numShiftVectors = detail::N_IVEC;

// from math/vectypes.h
constexpr const int dimX = 0; /* Defines for indexing in vectors */
constexpr const int dimY = 1;
constexpr const int dimZ = 2;

constexpr const int dimSize = 3;
typedef real        rvec[dimSize];
typedef real        matrix[dimSize][dimSize];
} // namespace nblib


#endif // NBLIB_BASICDEFINITIONS_H
