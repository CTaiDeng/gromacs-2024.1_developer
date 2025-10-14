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

#ifndef GMX_HARDWARE_DETECTHARDWARE_H
#define GMX_HARDWARE_DETECTHARDWARE_H

#include <memory>

#include "gromacs/utility/gmxmpi.h"

struct gmx_hw_info_t;

namespace gmx
{
class HardwareTopology;
class MDLogger;
class PhysicalNodeCommunicator;

/*! \brief Run detection and make correct and consistent
 * hardware information available on all ranks.
 *
 * May do communication on libraryCommWorld when compiled with real MPI.
 *
 * This routine is designed to be called once on each process.  In a
 * thread-MPI configuration, it may only be called before the threads
 * are spawned. With real MPI, communication is needed to coordinate
 * the results. In all cases, any thread within a process may use the
 * returned handle.
 */
std::unique_ptr<gmx_hw_info_t> gmx_detect_hardware(const PhysicalNodeCommunicator& physicalNodeComm,
                                                   MPI_Comm libraryCommWorld);

/*! \brief Issue warnings to mdlog that were decided during detection
 *
 * \param[in] mdlog                Logger
 * \param[in] hardwareInformation  The hardwareInformation */
void logHardwareDetectionWarnings(const gmx::MDLogger& mdlog, const gmx_hw_info_t& hardwareInformation);

} // namespace gmx

#endif
