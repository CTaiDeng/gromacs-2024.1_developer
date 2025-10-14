/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2020- The GROMACS Authors
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

/*! \internal
 * \brief Encapsulates membed methods
 *
 * \author Joe Jordan <ejjordan@kth.se>
 * \ingroup module_mdrun
 */
#include "gmxpre.h"

#include "membedholder.h"

#include "gromacs/commandline/filenm.h"
#include "gromacs/mdlib/membed.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/state.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/real.h"

namespace gmx
{

MembedHolder::MembedHolder(int nfile, const t_filenm fnm[]) :
    doMembed_(opt2bSet("-membed", nfile, fnm))
{
}

MembedHolder::~MembedHolder()
{
    if (doMembed_)
    {
        free_membed(membed_);
    }
}

void MembedHolder::initializeMembed(FILE*          fplog,
                                    int            nfile,
                                    const t_filenm fnm[],
                                    gmx_mtop_t*    mtop,
                                    t_inputrec*    inputrec,
                                    t_state*       state,
                                    t_commrec*     cr,
                                    real*          cpt)
{
    if (doMembed_)
    {
        if (MAIN(cr))
        {
            fprintf(stderr, "Initializing membed");
        }
        /* Note that membed cannot work in parallel because mtop is
         * changed here. Fix this if we ever want to make it run with
         * multiple ranks. */
        membed_ = init_membed(fplog, nfile, fnm, mtop, inputrec, state, cr, cpt);
    }
}

gmx_membed_t* MembedHolder::membed()
{
    return membed_;
}

MembedHolder::MembedHolder(MembedHolder&& holder) noexcept
{
    doMembed_        = holder.doMembed_;
    membed_          = holder.membed_;
    holder.membed_   = nullptr;
    holder.doMembed_ = false;
}

MembedHolder& MembedHolder::operator=(MembedHolder&& holder) noexcept
{
    if (&holder != this)
    {
        doMembed_        = holder.doMembed_;
        membed_          = holder.membed_;
        holder.membed_   = nullptr;
        holder.doMembed_ = false;
    }
    return *this;
}

} // namespace gmx
