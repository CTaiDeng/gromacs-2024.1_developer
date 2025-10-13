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
 * Declares helper functions for dumping basic data structures as text.
 *
 * \inlibraryapi
 * \ingroup module_utility
 */
#ifndef GMX_FILEIO_TXTDUMP_H
#define GMX_FILEIO_TXTDUMP_H

#include <cstdio>

#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/real.h"

//! Line width for text dump output.
#define LINE_WIDTH 80
//! Right margin for text dump output.
#define RMARGIN 10
//! Actual line length for text dump output.
#define USE_WIDTH ((LINE_WIDTH) - (RMARGIN))
//! Default indentation for text dump output.
#define INDENT 3

//! Prints an initial indentation for a line.
int pr_indent(FILE* fp, int n);
//! Returns whether \p is available (not null), and prints a note if it is not.
bool available(FILE* fp, const void* p, int indent, const char* title);
//! Prints a title for a dumped section.
int pr_title(FILE* fp, int indent, const char* title);
//! Prints a title for a dumped section with a number suffixed.
int pr_title_n(FILE* fp, int indent, const char* title, int n);
//! Prints a title for a dumped section with two numbers suffixed (in NxM format).
int pr_title_nxn(FILE* fp, int indent, const char* title, int n1, int n2);
//! Prints an array of reals.
void pr_reals(FILE* fp, int indent, const char* title, const real vec[], int n);
//! Prints an array of doubles.
void pr_doubles(FILE* fp, int indent, const char* title, const double* vec, int n);
//! Prints an array of reals as a matrix with inner dimension dim.
void pr_reals_of_dim(FILE* fp, int indent, const char* title, const real* vec, int n, int dim);
//! Prints an integer value.
void pr_int(FILE* fp, int indent, const char* title, int i);
//! Prints a int64_t value.
void pr_int64(FILE* fp, int indent, const char* title, int64_t i);
//! Prints a floating-point value.
void pr_real(FILE* fp, int indent, const char* title, real r);
//! Prints a double-precision floating-point value.
void pr_double(FILE* fp, int indent, const char* title, double d);
//! Prints a string value.
void pr_str(FILE* fp, int indent, const char* title, const char* s);
//! Prints strings as a section; intended to be used for an array of names.
void pr_strings(FILE* fp, int indent, const char* title, const char* const* const* nm, int n, gmx_bool bShowNumbers);

#endif
