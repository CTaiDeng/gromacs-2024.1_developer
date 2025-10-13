/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 1991- The GROMACS Authors
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

/*! \libinternal \file
 * \brief
 * Utilities for comparing data structures (for gmx check).
 *
 * \inlibraryapi
 * \ingroup module_utility
 */
#ifndef GMX_UTILITY_COMPARE_H
#define GMX_UTILITY_COMPARE_H

#include <cstdio>

#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/real.h"

//! Compares two real values for equality.
gmx_bool equal_real(real i1, real i2, real ftol, real abstol);
//! Compares two float values for equality.
gmx_bool equal_float(float i1, float i2, float ftol, float abstol);
//! Compares two double values for equality.
gmx_bool equal_double(double i1, double i2, real ftol, real abstol);

//! Compares two integers and prints differences.
void cmp_int(FILE* fp, const char* s, int index, int i1, int i2);

//! Compares two 64-bit integers and prints differences.
void cmp_int64(FILE* fp, const char* s, int64_t i1, int64_t i2);

//! Compares two unsigned short values and prints differences.
void cmp_us(FILE* fp, const char* s, int index, unsigned short i1, unsigned short i2);

//! Compares two unsigned char values and prints differences.
void cmp_uc(FILE* fp, const char* s, int index, unsigned char i1, unsigned char i2);

//! Compares two boolean values and prints differences, and returns whether both are true.
gmx_bool cmp_bool(FILE* fp, const char* s, int index, gmx_bool b1, gmx_bool b2);

//! Compares two strings and prints differences.
void cmp_str(FILE* fp, const char* s, int index, const char* s1, const char* s2);

//! Compares two reals and prints differences.
void cmp_real(FILE* fp, const char* s, int index, real i1, real i2, real ftol, real abstol);

//! Compares two floats and prints differences.
void cmp_float(FILE* fp, const char* s, int index, float i1, float i2, float ftol, float abstol);

//! Compares two doubles and prints differences.
void cmp_double(FILE* fp, const char* s, int index, double i1, double i2, double ftol, double abstol);

//! Compare two enums of generic type and print differences.
template<typename EnumType>
void cmpEnum(FILE* fp, const char* s, EnumType value1, EnumType value2)
{
    if (value1 != value2)
    {
        fprintf(fp, "%s (", s);
        fprintf(fp, "%s", enumValueToString(value1));
        fprintf(fp, " - ");
        fprintf(fp, "%s", enumValueToString(value2));
        fprintf(fp, ")\n");
    }
}

#endif
