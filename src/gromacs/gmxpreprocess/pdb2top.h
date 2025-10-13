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

#ifndef GMX_GMXPREPROCESS_PDB2TOP_H
#define GMX_GMXPREPROCESS_PDB2TOP_H

#include <cstdio>

#include <filesystem>
#include <string>
#include <vector>

#include "gromacs/math/vectypes.h"
#include "gromacs/utility/enumerationhelpers.h"

class PreprocessingAtomTypes;

namespace gmx
{
template<typename>
class ArrayRef;
class MDLogger;
} // namespace gmx

struct t_atoms;
struct t_excls;
struct MoleculePatchDatabase;
struct t_mols;
struct InteractionsOfType;
struct t_resinfo;
struct PreprocessResidue;
struct DisulfideBond;
struct t_symtab;

/* this *MUST* correspond to array in pdb2top.cpp */
enum class HistidineStates : int
{
    A,
    B,
    H,
    One,
    Count
};
const char* enumValueToString(HistidineStates enumValue);

std::filesystem::path choose_ff(const char* ffsel, char* forcefield, int ff_maxlen, const gmx::MDLogger& logger);
/* Find force fields in the current and libdirs and choose an ff.
 * If ffsel!=NULL: search for ffsel.
 * If ffsel==NULL: interactive selection.
 */

void choose_watermodel(const char*                  wmsel,
                       const std::filesystem::path& ffdir,
                       char**                       watermodel,
                       const gmx::MDLogger&         logger);
/* Choose, possibly interactively, which water model to include,
 * based on the wmsel command line option choice and watermodels.dat
 * in ffdir.
 */

void get_hackblocks_rtp(std::vector<MoleculePatchDatabase>*    globalPatches,
                        std::vector<PreprocessResidue>*        usedPpResidues,
                        gmx::ArrayRef<const PreprocessResidue> rtpFFDB,
                        int                                    nres,
                        t_resinfo*                             resinfo,
                        int                                    nterpairs,
                        t_symtab*                              symtab,
                        gmx::ArrayRef<MoleculePatchDatabase*>  ntdb,
                        gmx::ArrayRef<MoleculePatchDatabase*>  ctdb,
                        gmx::ArrayRef<const int>               rn,
                        gmx::ArrayRef<const int>               rc,
                        bool                                   bAllowMissing,
                        const gmx::MDLogger&                   logger);
/* Get the database entries for the nres residues in resinfo
 * and store them in restp and hb.
 */

void match_atomnames_with_rtp(gmx::ArrayRef<PreprocessResidue>     usedPpResidues,
                              gmx::ArrayRef<MoleculePatchDatabase> globalPatches,
                              t_atoms*                             pdba,
                              t_symtab*                            symtab,
                              gmx::ArrayRef<gmx::RVec>             x,
                              bool                                 bVerbose,
                              const gmx::MDLogger&                 logger);
/* Check if atom in pdba need to be deleted of renamed due to tdb or hdb.
 * If renaming involves atoms added wrt to the rtp database,
 * add these atoms to restp.
 */

void print_top_comment(FILE*                        out,
                       const std::filesystem::path& filename,
                       const std::filesystem::path& ffdir,
                       bool                         bITP);

void print_top_header(FILE*                        out,
                      const std::filesystem::path& filename,
                      bool                         bITP,
                      const std::filesystem::path& ffdir,
                      real                         mHmult);

void print_top_mols(FILE*                                      out,
                    const char*                                title,
                    const std::filesystem::path&               ffdir,
                    const char*                                water,
                    gmx::ArrayRef<const std::filesystem::path> incls,
                    gmx::ArrayRef<const t_mols>                mols);

void write_top(FILE*                                   out,
               const std::filesystem::path&            pr,
               const char*                             molname,
               t_atoms*                                at,
               bool                                    bRTPresname,
               gmx::ArrayRef<const int>                bts,
               gmx::ArrayRef<const InteractionsOfType> plist,
               t_excls                                 excls[],
               PreprocessingAtomTypes*                 atype,
               int*                                    cgnr,
               int                                     nrexcl);
/* NOTE: nrexcl is not the size of *excl! */

void pdb2top(FILE*                                  top_file,
             const std::filesystem::path&           posre_fn,
             const char*                            molname,
             t_atoms*                               atoms,
             std::vector<gmx::RVec>*                x,
             PreprocessingAtomTypes*                atype,
             t_symtab*                              tab,
             gmx::ArrayRef<const PreprocessResidue> rtpFFDB,
             gmx::ArrayRef<PreprocessResidue>       usedPpResidues,
             gmx::ArrayRef<MoleculePatchDatabase>   globalPatches,
             bool                                   bAllowMissing,
             bool                                   bVsites,
             bool                                   bVsiteAromatics,
             const std::filesystem::path&           ffdir,
             real                                   mHmult,
             gmx::ArrayRef<const DisulfideBond>     ssbonds,
             real                                   long_bond_dist,
             real                                   short_bond_dist,
             bool                                   bDeuterate,
             bool                                   bChargeGroups,
             bool                                   bCmap,
             bool                                   bRenumRes,
             bool                                   bRTPresname,
             gmx::ArrayRef<const int>               cyclicBondsIndex,
             const gmx::MDLogger&                   logger);
/* Create a topology ! */

void print_sums(const t_atoms* atoms, bool bSystem, const gmx::MDLogger& logger);

#endif
