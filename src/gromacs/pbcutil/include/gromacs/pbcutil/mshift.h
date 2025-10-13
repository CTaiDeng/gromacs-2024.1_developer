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

#ifndef GMX_PBCUTIL_MSHIFT_H
#define GMX_PBCUTIL_MSHIFT_H

#include <cstdio>

#include <vector>

#include "gromacs/math/vectypes.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/listoflists.h"

struct InteractionList;
struct gmx_moltype_t;
class InteractionDefinitions;
struct t_idef;
enum class PbcType : int;

typedef enum
{
    egcolWhite,
    egcolGrey,
    egcolBlack,
    egcolNR
} egCol;

/* Struct used to make molecules broken over PBC whole
 *
 * TODO: Should be turned into a proper class
 */
struct t_graph
{
    /* Describes the connectivity between, potentially, multiple parts of
     * the ilist that are internally chemically bonded together.
     */
    enum class BondedParts
    {
        Single,               /* All atoms are connected through chemical bonds */
        MultipleDisconnected, /* There are multiple parts, e.g. monomers, that are all disconnected */
        MultipleConnected /* There are multiple parts, e.g. monomers, that are partially or fully connected between each other by interactions other than chemical bonds */
    };

    // Returns the number of nodes stored in the graph (can be less than shiftAtomEnd)
    int numNodes() const { return edges.size(); }

    // Shift atoms up to shiftAtomEnd
    int shiftAtomEnd = 0;
    // The number of atoms that are connected to other atoms in the graph
    int numConnectedAtoms = 0;
    // The first connected atom in the graph
    int edgeAtomBegin = 0;
    // The last connected atom in the graph
    int edgeAtomEnd = 0;
    //  The graph: list of atoms connected to each atom, indexing is offset by -edgeAtomBegin
    gmx::ListOfLists<int> edges;
    // Whether we are using screw PBC
    bool useScrewPbc = false;
    // Shift for each particle, updated after putting atoms in the box
    std::vector<gmx::IVec> ishift;
    // Work buffer for coloring nodes
    std::vector<egCol> edgeColor;
    // Tells how connected this graph is
    BondedParts parts = BondedParts::Single;
};

#define SHIFT_IVEC(g, i) ((g)->ishift[i])

t_graph mk_graph(const InteractionDefinitions& idef, int numAtoms);
/* Build a graph from an idef description. The graph can be used
 * to generate mol-shift indices.
 * numAtoms should coincide will molecule boundaries,
 */

t_graph* mk_graph(FILE*                         fplog,
                  const InteractionDefinitions& idef,
                  int                           shiftAtomEnd,
                  gmx_bool                      bShakeOnly,
                  gmx_bool                      bSettle);
/* Build a graph from an idef description. The graph can be used
 * to generate mol-shift indices.
 * Shifts atoms up to shiftAtomEnd, which should coincide with a molecule boundary,
 * for the whole system this is simply natoms.
 * If bShakeOnly, only the connections in the shake list are used.
 * If bSettle && bShakeOnly the settles are used too.
 */

t_graph* mk_graph(FILE* fplog, const struct t_idef* idef, int shiftAtomEnd, gmx_bool bShakeOnly, gmx_bool bSettle);
/* Build a graph from an idef description. The graph can be used
 * to generate mol-shift indices.
 * Shifts atoms up to shiftAtomEnd, which should coincide with a molecule boundary,
 * for the whole system this is simply natoms.
 * If bShakeOnly, only the connections in the shake list are used.
 * If bSettle && bShakeOnly the settles are used too.
 */

t_graph mk_graph_moltype(const gmx_moltype_t& moltype);
/* As mk_graph, but takes gmx_moltype_t iso t_idef */


void done_graph(t_graph* g);
/* Free the memory in *g and the pointer g */

void p_graph(FILE* log, const char* title, const t_graph* g);
/* Print a graph to log */

void mk_mshift(FILE* log, t_graph* g, PbcType pbcType, const matrix box, const rvec x[]);
/* Calculate the mshift codes, based on the connection graph in g. */

void shift_x(const t_graph* g, const matrix box, const rvec x[], rvec x_s[]);
/* Add the shift vector to x, and store in x_s (may be same array as x) */

void shift_self(const t_graph& g, const matrix box, rvec x[]);
/* Id. but in place */

void shift_self(const t_graph* g, const matrix box, rvec x[]);
/* Id. but in place */

void unshift_x(const t_graph* g, const matrix box, rvec x[], const rvec x_s[]);
/* Subtract the shift vector from x_s, and store in x (may be same array) */

void unshift_self(const t_graph* g, const matrix box, rvec x[]);
/* Id, but in place */

#endif
