/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2022- The GROMACS Authors
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
 *  \brief Lookup Intel hardware version from PCI Express ID.
 *
 *  Extracted into a separate file because it contains a huge data table.
 *
 *  \author Andrey Alekseenko <al42and@gmail.com>
 *
 * \internal
 * \ingroup module_hardware
 */
#ifndef GMX_HARDWARE_DEVICE_MANAGEMENT_SYCL_INTEL_DEVICE_IDS_H
#define GMX_HARDWARE_DEVICE_MANAGEMENT_SYCL_INTEL_DEVICE_IDS_H

#include <optional>
#include <tuple>

/*! \brief Look up Intel hardware version from device's PCI Express ID.
 *
 * The returned values correspond to the ones \c ocloc uses.
 *
 * \param[in] pciExpressID Device ID reported in the device name.
 * \returns major.minor.revision if device is found in the database, \c std::nullopt otherwise.
 */
std::optional<std::tuple<int, int, int>> getIntelHardwareVersionFromPciExpressID(unsigned int pciExpressID);

#endif // GMX_HARDWARE_DEVICE_MANAGEMENT_SYCL_INTEL_DEVICE_IDS_H
