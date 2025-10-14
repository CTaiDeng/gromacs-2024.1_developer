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

#ifndef GMX_GMXPREPROCESS_TOPIO_H
#define GMX_GMXPREPROCESS_TOPIO_H

#include <memory>
#include <vector>

#include "gromacs/utility/real.h"

struct gmx_molblock_t;
struct gmx_mtop_t;
class PreprocessingAtomTypes;
struct t_gromppopts;
struct t_inputrec;
struct MoleculeInformation;
struct InteractionsOfType;
struct t_symtab;
class WarningHandler;
enum class CombinationRule : int;

namespace gmx
{
template<typename>
class ArrayRef;
class MDLogger;
} // namespace gmx

double check_mol(const gmx_mtop_t* mtop, WarningHandler* wi);
/* Check mass and charge */

char** do_top(bool                                  bVerbose,
              const char*                           topfile,
              const char*                           topppfile,
              t_gromppopts*                         opts,
              bool                                  bZero,
              t_symtab*                             symtab,
              gmx::ArrayRef<InteractionsOfType>     plist,
              CombinationRule*                      combination_rule,
              double*                               repulsion_power,
              real*                                 fudgeQQ,
              PreprocessingAtomTypes*               atype,
              std::vector<MoleculeInformation>*     molinfo,
              std::unique_ptr<MoleculeInformation>* intermolecular_interactions,
              const t_inputrec*                     ir,
              std::vector<gmx_molblock_t>*          molblock,
              bool*                                 ffParametrizedWithHBondConstraints,
              WarningHandler*                       wi,
              const gmx::MDLogger&                  logger);

/* This routine expects sys->molt[m].ilist to be of size F_NRE and ordered. */
void generate_qmexcl(gmx_mtop_t* sys, t_inputrec* ir, const gmx::MDLogger& logger);

#endif
