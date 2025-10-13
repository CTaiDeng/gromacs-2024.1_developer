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

#ifndef GMX_GMXPREPROCESS_TOPPUSH_H
#define GMX_GMXPREPROCESS_TOPPUSH_H

#include <vector>

#include "gromacs/utility/real.h"

enum class Directive : int;
class PreprocessingAtomTypes;
class PreprocessingBondAtomType;
struct t_atoms;
struct t_block;
struct MoleculeInformation;
struct t_nbparam;
class InteractionOfType;
struct InteractionsOfType;
struct PreprocessResidue;
class WarningHandler;
enum class CombinationRule : int;
namespace gmx
{
template<typename>
class ArrayRef;
struct ExclusionBlock;
} // namespace gmx

void generate_nbparams(CombinationRule         comb,
                       int                     funct,
                       InteractionsOfType*     plist,
                       PreprocessingAtomTypes* atype,
                       WarningHandler*         wi);

void push_at(PreprocessingAtomTypes*    at,
             PreprocessingBondAtomType* bat,
             char*                      line,
             int                        nb_funct,
             t_nbparam***               nbparam,
             t_nbparam***               pair,
             WarningHandler*            wi);

void push_bt(Directive                         d,
             gmx::ArrayRef<InteractionsOfType> bt,
             int                               nral,
             PreprocessingAtomTypes*           at,
             PreprocessingBondAtomType*        bat,
             char*                             line,
             WarningHandler*                   wi);

void push_dihedraltype(Directive                         d,
                       gmx::ArrayRef<InteractionsOfType> bt,
                       PreprocessingBondAtomType*        bat,
                       char*                             line,
                       WarningHandler*                   wi);

void push_cmaptype(Directive                         d,
                   gmx::ArrayRef<InteractionsOfType> bt,
                   int                               nral,
                   PreprocessingAtomTypes*           at,
                   PreprocessingBondAtomType*        bat,
                   char*                             line,
                   WarningHandler*                   wi);

void push_nbt(Directive d, t_nbparam** nbt, PreprocessingAtomTypes* atype, char* plines, int nb_funct, WarningHandler* wi);

void push_atom(struct t_symtab* symtab, t_atoms* at, PreprocessingAtomTypes* atype, char* line, WarningHandler* wi);

void push_bond(Directive                         d,
               gmx::ArrayRef<InteractionsOfType> bondtype,
               gmx::ArrayRef<InteractionsOfType> bond,
               t_atoms*                          at,
               PreprocessingAtomTypes*           atype,
               char*                             line,
               bool                              bBonded,
               bool                              bGenPairs,
               real                              fudgeQQ,
               bool                              bZero,
               bool*                             bWarn_copy_A_B,
               WarningHandler*                   wi);

void push_cmap(Directive                         d,
               gmx::ArrayRef<InteractionsOfType> bondtype,
               gmx::ArrayRef<InteractionsOfType> bond,
               t_atoms*                          at,
               PreprocessingAtomTypes*           atype,
               char*                             line,
               WarningHandler*                   wi);

void push_vsitesn(Directive d, gmx::ArrayRef<InteractionsOfType> bond, t_atoms* at, char* line, WarningHandler* wi);

void push_mol(gmx::ArrayRef<MoleculeInformation> mols, char* pline, int* whichmol, int* nrcopies, WarningHandler* wi);

void push_molt(struct t_symtab* symtab, std::vector<MoleculeInformation>* mol, char* line, WarningHandler* wi);

void push_excl(char* line, gmx::ArrayRef<gmx::ExclusionBlock> b2, WarningHandler* wi);

int copy_nbparams(t_nbparam** param, int ftype, InteractionsOfType* plist, int nr);

void free_nbparam(t_nbparam** param, int nr);

int add_atomtype_decoupled(PreprocessingAtomTypes* at, t_nbparam*** nbparam, t_nbparam*** pair);
/* Add an atom type with all parameters set to zero (no interactions).
 * Returns the atom type number.
 */

void convert_moltype_couple(MoleculeInformation* mol,
                            int                  atomtype_decouple,
                            real                 fudgeQQ,
                            int                  couple_lam0,
                            int                  couple_lam1,
                            bool                 bCoupleIntra,
                            int                  nb_funct,
                            InteractionsOfType*  nbp,
                            WarningHandler*      wi);
/* Setup mol such that the B-state has no interaction with the rest
 * of the system, but full interaction with itself.
 */

#endif
