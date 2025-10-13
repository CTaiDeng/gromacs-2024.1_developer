/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2015- The GROMACS Authors
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

#include "gmxpre.h"

#include "gromacs/utility/int64_to_int.h"

#include <cstdio>

#include "gromacs/utility/basedefinitions.h"

int int64_to_int(int64_t step, const char* warn)
{
    int i = static_cast<int>(step);

    if (warn != nullptr && (static_cast<int64_t>(i) != step))
    {
        fprintf(stderr, "\nWARNING during %s:\n", warn);
        fprintf(stderr, "int64 value ");
        fprintf(stderr, "%" PRId64, step);
        fprintf(stderr, " does not fit in int, converted to %d\n\n", i);
    }

    return i;
}
