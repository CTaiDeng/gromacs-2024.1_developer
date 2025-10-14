/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 1991- The GROMACS Authors
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

#ifndef GMX_GMXPREPROCESS_GPP_NEXTNB_H
#define GMX_GMXPREPROCESS_GPP_NEXTNB_H

struct InteractionsOfType;

namespace gmx
{
template<typename>
class ArrayRef;
template<typename>
class ListOfLists;
} // namespace gmx

struct t_nextnb
{
    int nr;   /* nr atoms (0 <= i < nr) (atoms->nr)	        */
    int nrex; /* with nrex lists of neighbours		*/
    /* respectively containing zeroth, first	*/
    /* second etc. neigbours (0 <= nre < nrex)	*/
    int** nrexcl; /* with (0 <= nrx < nrexcl[i][nre]) neigbours    */
    /* per list stored in one 2d array of lists	*/
    int*** a; /* like this: a[i][nre][nrx]			*/
};

void init_nnb(t_nextnb* nnb, int nr, int nrex);
/* Initiate the arrays for nnb (see above) */

void done_nnb(t_nextnb* nnb);
/* Cleanup the nnb struct */

#ifdef DEBUG_NNB
#    define print_nnb(nnb, s) __print_nnb(nnb, s)
void print_nnb(t_nextnb* nnb, char* s);
/* Print the nnb struct */
#else
#    define print_nnb(nnb, s)
#endif

void gen_nnb(t_nextnb* nnb, gmx::ArrayRef<InteractionsOfType> plist);
/* Generate a t_nextnb structure from bond information.
 * With the structure you can either generate exclusions
 * or generate angles and dihedrals. The structure must be
 * initiated using init_nnb.
 */

void generate_excl(int nrexcl, int nratoms, gmx::ArrayRef<InteractionsOfType> plist, gmx::ListOfLists<int>* excls);
/* Generate an exclusion block from bonds and constraints in
 * plist.
 */

#endif
