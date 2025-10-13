/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2013- The GROMACS Authors
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

/*! \libinternal \file
 *
 * \brief
 * Declares functions needed for reading, initializing and setting the AWH parameter data types.
 *
 * \author Viveca Lindahl
 * \inlibraryapi
 * \ingroup module_awh
 */

#ifndef GMX_AWH_READPARAMS_H
#define GMX_AWH_READPARAMS_H

#include "gromacs/fileio/readinp.h"
#include "gromacs/math/vectypes.h"

struct t_grpopts;
struct t_inputrec;
struct gmx_mtop_t;
struct pull_params_t;
struct pull_t;
enum class PbcType : int;

namespace gmx
{

class AwhParams;
class ISerializer;

/*! \brief Check the AWH parameters.
 *
 * \param[in]     awhParams    The AWH parameters.
 * \param[in]     inputrec     Input parameter struct.
 * \param[in,out] wi           Struct for bookeeping warnings.
 */
void checkAwhParams(const AwhParams& awhParams, const t_inputrec& inputrec, WarningHandler* wi);


/*! \brief
 * Sets AWH parameters that need state parameters such as the box vectors.
 *
 * \param[in,out] awhParams        AWH parameters.
 * \param[in]     pull_params      Pull parameters.
 * \param[in,out] pull_work        Pull working struct to register AWH bias in.
 * \param[in]     box              Box vectors.
 * \param[in]     pbcType          Periodic boundary conditions enum.
 * \param[in]     compressibility  Compressibility matrix for pressure coupling, pass all 0
 *                                 without pressure coupling
 * \param[in]     inputrec         Input record, for checking the reference temperature
 * \param[in]     initLambda       The starting lambda, to allow using free energy lambda
 *                                 as reaction coordinate provider in any dimension.
 * \param[in]     mtop             The system topology.
 * \param[in,out] wi               Struct for bookeeping warnings.
 *
 * \note This function currently relies on the function set_pull_init to have been called.
 */
void setStateDependentAwhParams(AwhParams*           awhParams,
                                const pull_params_t& pull_params,
                                pull_t*              pull_work,
                                const matrix         box,
                                PbcType              pbcType,
                                const tensor&        compressibility,
                                const t_inputrec&    inputrec,
                                real                 initLambda,
                                const gmx_mtop_t&    mtop,
                                WarningHandler*      wi);

//! Returns true when AWH has a bias with a free energy lambda state dimension
bool awhHasFepLambdaDimension(const AwhParams& awhParams);

} // namespace gmx

#endif /* GMX_AWH_READPARAMS_H */
