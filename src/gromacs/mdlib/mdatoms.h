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

#ifndef GMX_MDLIB_MDATOMS_H
#define GMX_MDLIB_MDATOMS_H

#include <cstdio>

#include <memory>
#include <vector>

#include "gromacs/gpu_utils/hostallocator.h"
#include "gromacs/utility/real.h"
#include "gromacs/utility/unique_cptr.h"

struct gmx_mtop_t;
struct t_inputrec;
struct t_mdatoms;

namespace gmx
{

/*! \libinternal
 * \brief Contains a C-style t_mdatoms while managing some of its
 * memory with C++ vectors with allocators.
 *
 * \todo The group-scheme kernels needed a plain C-style t_mdatoms, so
 * this type combines that with the memory management needed for
 * efficient PME on GPU transfers. The mdatoms_ member should be
 * removed. */
class MDAtoms
{
    //! C-style mdatoms struct.
    std::unique_ptr<t_mdatoms> mdatoms_;
    //! Memory for chargeA that can be set up for efficient GPU transfer.
    gmx::PaddedHostVector<real> chargeA_;
    //! Memory for chargeB that can be set up for efficient GPU transfer.
    gmx::PaddedHostVector<real> chargeB_;

public:
    // TODO make this private
    MDAtoms();
    //! Getter.
    t_mdatoms* mdatoms() { return mdatoms_.get(); }
    //! Const getter.
    const t_mdatoms* mdatoms() const { return mdatoms_.get(); }
    /*! \brief Resizes memory for charges of FEP state A.
     *
     * \throws std::bad_alloc  If out of memory.
     */
    void resizeChargeA(int newSize);
    /*! \brief Resizes memory for charges of FEP state B.
     *
     * \throws std::bad_alloc  If out of memory.
     */
    void resizeChargeB(int newSize);
    //! Builder function.
    friend std::unique_ptr<MDAtoms>
    makeMDAtoms(FILE* fp, const gmx_mtop_t& mtop, const t_inputrec& ir, bool rankHasPmeGpuTask);
};

//! Builder function for MdAtomsWrapper.
std::unique_ptr<MDAtoms> makeMDAtoms(FILE* fp, const gmx_mtop_t& mtop, const t_inputrec& ir, bool useGpuForPme);

} // namespace gmx

/*! \brief This routine copies the atoms->atom struct into md.
 *
 * \param[in]    mtop     The molecular topology.
 * \param[in]    inputrec The input record.
 * \param[in]    nindex   If nindex>=0 we are doing DD.
 * \param[in]    index    Lookup table for global atom index.
 * \param[in]    homenr   Number of atoms on this processor.
 * \param[inout] mdAtoms  Data set up by this routine.
 *
 * If index!=NULL only the indexed atoms are copied.
 * For the masses the A-state (lambda=0) mass is used.
 * Sets md->lambda = 0.
 * In free-energy runs, update_mdatoms() should be called after atoms2md()
 * to set the masses corresponding to the value of lambda at each step.
 */
void atoms2md(const gmx_mtop_t&  mtop,
              const t_inputrec&  inputrec,
              int                nindex,
              gmx::ArrayRef<int> index,
              int                homenr,
              gmx::MDAtoms*      mdAtoms);

void update_mdatoms(t_mdatoms* md, real lambda);
/* When necessary, sets all the mass parameters to values corresponding
 * to the free-energy parameter lambda.
 * Sets md->lambda = lambda.
 */

#endif
