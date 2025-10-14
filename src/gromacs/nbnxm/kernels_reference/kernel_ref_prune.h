/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2012- The GROMACS Authors
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
 *
 * \brief
 * Declares the C reference pruning only kernel.
 *
 * \author Berk Hess <hess@kth.se>
 * \ingroup module_nbnxm
 */

#include "gromacs/math/vectypes.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/real.h"

struct nbnxn_atomdata_t;
struct NbnxnPairlistCpu;

namespace gmx
{
template<typename>
class ArrayRef;
}
/*! \brief Prune a single NbnxnPairlistCpu entry with distance \p rlistInner
 *
 * Reads a cluster pairlist \p nbl->ciOuter, \p nbl->cjOuter and writes
 * all cluster pairs within \p rlistInner to \p nbl->ci, \p nbl->cj.
 */
void nbnxn_kernel_prune_ref(NbnxnPairlistCpu*              nbl,
                            const nbnxn_atomdata_t*        nbat,
                            gmx::ArrayRef<const gmx::RVec> shiftvec,
                            real                           rlistInner);
