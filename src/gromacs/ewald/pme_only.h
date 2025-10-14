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

/*! \libinternal \file
 *
 * \brief This file contains function declarations necessary for
 * running on an MPI rank doing only PME long-ranged work.
 *
 * \author Berk Hess <hess@kth.se>
 * \inlibraryapi
 * \ingroup module_ewald
 */

#ifndef GMX_EWALD_PME_ONLY_H
#define GMX_EWALD_PME_ONLY_H

#include <string>

#include "gromacs/timing/walltime_accounting.h"

struct t_commrec;
struct t_inputrec;
struct t_nrnb;
struct gmx_pme_t;
struct gmx_wallcycle;

enum class PmeRunMode;
namespace gmx
{
class DeviceStreamManager;
}

/*! \brief Called on the nodes that do PME exclusively */
int gmx_pmeonly(gmx_pme_t**                     pme,
                const t_commrec*                cr,
                t_nrnb*                         mynrnb,
                gmx_wallcycle*                  wcycle,
                gmx_walltime_accounting_t       walltime_accounting,
                t_inputrec*                     ir,
                PmeRunMode                      runMode,
                bool                            useGpuPmePpCommunication,
                bool                            useNvshmem,
                const gmx::DeviceStreamManager* deviceStreamManager);

#endif
