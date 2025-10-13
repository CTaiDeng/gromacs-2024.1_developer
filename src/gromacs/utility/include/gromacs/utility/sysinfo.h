/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2014- The GROMACS Authors
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
 * \brief
 * Declares functions that wrap platform-specific calls for obtaining
 * information about the operating environment and the current
 * process.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \inlibraryapi
 * \ingroup module_utility
 */
#ifndef GMX_UTILITY_SYSINFO_H
#define GMX_UTILITY_SYSINFO_H

#include <cstddef>
#include <ctime>

#include <string>

/*! \addtogroup module_utility
 * \{
 */

/*! \brief
 * Gets the hostname as given by gethostname(), if available.
 *
 * \param[out] buf  Buffer to receive the hostname.
 * \param[in]  len  Length of buffer \p buf (must be >= 8).
 * \returns 0 on success, -1 on error.
 *
 * If the value is not available, "unknown" is returned.
 * \p name should have at least size \p len.
 *
 * Does not throw.
 */
int gmx_gethostname(char* buf, size_t len);

/*! \brief
 * Returns the process ID of the current process.
 *
 * Does not throw.
 */
int gmx_getpid();
/*! \brief
 * Returns the current user ID, or -1 if not available.
 *
 * Does not throw.
 */
int gmx_getuid();
/*! \brief
 * Gets the current user name, if available.
 *
 * \param[out] buf  Buffer to receive the username.
 * \param[in]  len  Length of buffer \p buf (must be >= 8).
 * \returns 0 on success, -1 on error.
 *
 * Does not throw.
 */
int gmx_getusername(char* buf, size_t len);

/*! \brief
 * Portable version of ctime_r.
 *
 * \throws std::bad_alloc when out of memory.
 */
std::string gmx_ctime_r(const time_t* clock);
/*! \brief
 * Gets the current time as a string.
 *
 * \throws std::bad_alloc when out of memory.
 */
std::string gmx_format_current_time();

/*! \brief
 * Wrapper for nice().
 *
 * Does not throw.
 */
int gmx_set_nice(int level);

/*! \} */

#endif
