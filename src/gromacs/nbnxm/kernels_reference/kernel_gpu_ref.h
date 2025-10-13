/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2012- The GROMACS Authors
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

/*! \internal \file
 *
 * \brief
 * Declares GPU reference kernel
 *
 * \author Berk Hess <hess@kth.se>
 * \ingroup module_nbnxm
 */

#ifndef GMX_NBNXM_KERNELS_REFERENCE_KERNEL_GPU_REF_H
#define GMX_NBNXM_KERNELS_REFERENCE_KERNEL_GPU_REF_H

#include "gromacs/math/vectypes.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/real.h"

struct NbnxnPairlistGpu;
struct nbnxn_atomdata_t;
struct interaction_const_t;
struct t_forcerec;

namespace gmx
{
class StepWorkload;
}

//! Reference (slow) kernel for nb n vs n GPU type pair lists
void nbnxn_kernel_gpu_ref(const NbnxnPairlistGpu*        nbl,
                          const nbnxn_atomdata_t*        nbat,
                          const interaction_const_t*     iconst,
                          gmx::ArrayRef<const gmx::RVec> shiftvec,
                          const gmx::StepWorkload&       stepWork,
                          int                            clearF,
                          gmx::ArrayRef<real>            f,
                          real*                          fshift,
                          real*                          Vc,
                          real*                          Vvdw);

#endif
