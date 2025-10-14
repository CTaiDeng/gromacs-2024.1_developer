/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2009- The GROMACS Authors
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

/*! \internal \file
 * \brief
 * Implements functions in selvalue.h.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_selection
 */
#include "gmxpre.h"

#include "selvalue.h"

#include "gromacs/selection/indexutil.h"
#include "gromacs/selection/position.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/smalloc.h"

void _gmx_selvalue_clear(gmx_ana_selvalue_t* val)
{
    val->nr     = 0;
    val->u.ptr  = nullptr;
    val->nalloc = 0;
}

void _gmx_selvalue_free(gmx_ana_selvalue_t* val)
{
    if (val->nalloc > 0)
    {
        if (val->type == POS_VALUE)
        {
            delete[] val->u.p;
        }
        else
        {
            sfree(val->u.ptr);
        }
    }
    // TODO: It causes a memory leak somewhere if val->nr is assigned zero here...
    val->u.ptr  = nullptr;
    val->nalloc = 0;
}

void _gmx_selvalue_reserve(gmx_ana_selvalue_t* val, int n)
{
    int i;

    if (val->nalloc == -1)
    {
        return;
    }

    if (!val->u.ptr || val->nalloc < n)
    {
        switch (val->type)
        {
            case INT_VALUE: srenew(val->u.i, n); break;
            case REAL_VALUE: srenew(val->u.r, n); break;
            case STR_VALUE:
                srenew(val->u.s, n);
                for (i = val->nalloc; i < n; ++i)
                {
                    val->u.s[i] = nullptr;
                }
                break;
            case POS_VALUE:
                GMX_RELEASE_ASSERT(val->u.ptr == nullptr,
                                   "Reallocation of position values not supported");
                val->u.p = new gmx_ana_pos_t[n];
                break;
            case GROUP_VALUE:
                srenew(val->u.g, n);
                for (i = val->nalloc; i < n; ++i)
                {
                    gmx_ana_index_clear(&val->u.g[i]);
                }
                break;
            case NO_VALUE: break;
        }
        val->nalloc = n;
    }
}

void _gmx_selvalue_getstore_and_release(gmx_ana_selvalue_t* val, void** ptr, int* nalloc)
{
    *ptr        = val->u.ptr;
    *nalloc     = val->nalloc;
    val->u.ptr  = nullptr;
    val->nalloc = 0;
}

void _gmx_selvalue_setstore(gmx_ana_selvalue_t* val, void* ptr)
{
    GMX_ASSERT(val->nalloc <= 0, "Memory leak from discarding an existing value");
    val->u.ptr  = ptr;
    val->nalloc = (ptr ? -1 : 0);
}

void _gmx_selvalue_setstore_alloc(gmx_ana_selvalue_t* val, void* ptr, int nalloc)
{
    GMX_ASSERT(val->nalloc <= 0 || (ptr == val->u.ptr && nalloc == val->nalloc),
               "Memory leak from discarding an existing value");
    val->u.ptr  = ptr;
    val->nalloc = nalloc;
}
