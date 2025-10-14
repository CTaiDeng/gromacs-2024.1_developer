/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2017- The GROMACS Authors
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

/*! \defgroup module_taskassignment Assigning simulation tasks to hardware (taskassignment)
 * \ingroup group_mdrun
 * \brief Provides code that manages assignment of simulation tasks to hardware.
 */
/*! \internal
 * \file
 * \brief Declares routine for reporting GPU usage.
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \ingroup module_taskassignment
 */
#ifndef GMX_TASKASSIGNMENT_REPORTGPUUSAGE_H
#define GMX_TASKASSIGNMENT_REPORTGPUUSAGE_H

#include <cstdlib>

#include <vector>

enum class PmeRunMode;

namespace gmx
{

class MDLogger;
struct GpuTaskMapping;
template<typename T>
class ArrayRef;
using GpuTaskAssignment = std::vector<GpuTaskMapping>;
class SimulationWorkload;

/*! \brief Log a report on how GPUs are being used on
 * the ranks of the physical node of rank 0 of the simulation.
 *
 * \todo It could be useful to report also whether any nodes differed,
 * and in what way.
 *
 * \param[in]  mdlog                               Logging object.
 * \param[in]  gpuTaskAssignmentOnRanksOfThisNode  The selected GPU IDs.
 * \param[in]  numGpuTasksOnThisNode               The number of GPU tasks on this node.
 * \param[in]  numPpRanks                          Number of PP ranks on this node
 * \param[in]  printHostName                       Print the hostname in the usage information
 * \param[in]  pmeRunMode                          Describes the execution of PME tasks
 * \param[in]  simulationWork                      Simulation workload descriptor
 *
 * \throws     std::bad_alloc if out of memory */
void reportGpuUsage(const MDLogger&                   mdlog,
                    ArrayRef<const GpuTaskAssignment> gpuTaskAssignmentOnRanksOfThisNode,
                    size_t                            numGpuTasksOnThisNode,
                    size_t                            numPpRanks,
                    bool                              printHostName,
                    PmeRunMode                        pmeRunMode,
                    const SimulationWorkload&         simulationWork);

} // namespace gmx

#endif
