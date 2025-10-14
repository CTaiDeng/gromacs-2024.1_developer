/*
 * Copyright (C) 2025 GaoZheng
 * SPDX-License-Identifier: GPL-3.0-only
 * This file is part of this project.
 * Licensed under the GNU General Public License version 3.
 * See https://www.gnu.org/licenses/gpl-3.0.html for details.
 */
/* This code is part of the tng compression routines.
 *
 * Written by Daniel Spangberg
 * Copyright (c) 2010, 2013, The GROMACS development team.
 *
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Revised BSD License.
 */


#ifndef BWT_H
#define BWT_H

void Ptngc_comp_to_bwt(unsigned int *vals, const int nvals,
		 unsigned int *output, int *index);

void Ptngc_comp_from_bwt(unsigned int *input, const int nvals, int index,
		   unsigned int *vals);

void Ptngc_bwt_merge_sort_inner(int *indices, const int nvals, unsigned int *vals,
                                const int start, const int end,
                                unsigned int *nrepeat,
                                int *workarray);

#endif
