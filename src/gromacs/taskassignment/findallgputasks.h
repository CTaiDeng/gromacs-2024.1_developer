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

/*! \internal
 * \file
 * \brief Declares routine for collecting all GPU tasks found on ranks of a node.
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \ingroup module_taskassignment
 */
#ifndef GMX_TASKASSIGNMENT_FINDALLGPUTASKS_H
#define GMX_TASKASSIGNMENT_FINDALLGPUTASKS_H

#include <vector>

namespace gmx
{

enum class GpuTask;
enum class TaskTarget;
class PhysicalNodeCommunicator;
template<typename T>
class ArrayRef;
//! Container of compute tasks suitable to run on a GPU e.g. on each rank of a node.
using GpuTasksOnRanks = std::vector<std::vector<GpuTask>>;

/*! \brief Returns container of all tasks on this rank
 * that are eligible for GPU execution.
 *
 * \param[in]  haveGpusOnThisPhysicalNode Whether there are any GPUs on this physical node.
 * \param[in]  nonbondedTarget            The user's choice for mdrun -nb for where to assign
 *                                        short-ranged nonbonded interaction tasks.
 * \param[in]  pmeTarget                  The user's choice for mdrun -pme for where to assign
 *                                        long-ranged PME nonbonded interaction tasks.
 * \param[in]  bondedTarget               The user's choice for mdrun -bonded for where to assign tasks.
 * \param[in]  updateTarget               The user's choice for mdrun -update for where to assign tasks.
 * \param[in]  useGpuForNonbonded         Whether GPUs will be used for nonbonded interactions.
 * \param[in]  useGpuForPme               Whether GPUs will be used for PME interactions.
 * \param[in]  rankHasPpTask              Whether this rank has a PP task
 * \param[in]  rankHasPmeTask             Whether this rank has a PME task
 */
std::vector<GpuTask> findGpuTasksOnThisRank(bool       haveGpusOnThisPhysicalNode,
                                            TaskTarget nonbondedTarget,
                                            TaskTarget pmeTarget,
                                            TaskTarget bondedTarget,
                                            TaskTarget updateTarget,
                                            bool       useGpuForNonbonded,
                                            bool       useGpuForPme,
                                            bool       rankHasPpTask,
                                            bool       rankHasPmeTask);

/*! \brief Returns container of all tasks on all ranks of this node
 * that are eligible for GPU execution.
 *
 * Perform all necessary communication for preparing for task
 * assignment. Separating this aspect makes it possible to unit test
 * the logic of task assignment. */
GpuTasksOnRanks findAllGpuTasksOnThisNode(ArrayRef<const GpuTask>         gpuTasksOnThisRank,
                                          const PhysicalNodeCommunicator& physicalNodeComm);

} // namespace gmx

#endif
