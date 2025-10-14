/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2010- The GROMACS Authors
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

/*! \file
 * \brief
 * Declares error codes and related functions for fatal error handling.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_utility
 */
#ifndef GMX_UTILITY_ERRORCODES_H
#define GMX_UTILITY_ERRORCODES_H

namespace gmx
{

/*! \addtogroup module_utility
 * \{
 */

/*! \brief
 * Possible error return codes from Gromacs functions.
 */
enum ErrorCode
{
    //! Zero for successful return.
    eeOK,

    //! Not enough memory to complete operation.
    eeOutOfMemory,
    //! Provided file could not be opened.
    eeFileNotFound,
    //! System I/O error.
    eeFileIO,
    //! Invalid user input (could not be understood).
    eeInvalidInput,
    //! Invalid user input (conflicting or unsupported settings).
    eeInconsistentInput,
    //! Requested tolerance cannot be achieved.
    eeTolerance,
    //! Simulation instability detected.
    eeInstability,

    /*! \name Error codes for buggy code
     *
     * Error codes below are for internal error checking; if triggered, they
     * should indicate a bug in the code.
     * \{
     */
    //! Requested feature not yet implemented.
    eeNotImplemented,
    //! Input value violates API specification.
    eeInvalidValue,
    //! Invalid routine called or wrong calling sequence detected.
    eeInvalidCall,
    //! Internal consistency check failed.
    eeInternalError,
    //! API specification was violated.
    eeAPIError,
    //! Range consistency check failed.
    eeRange,

    //! Parallel consistency check failed.
    eeParallelConsistency,

    //! Error specific for modular simulator.
    eeModularSimulator,
    //!\}

    //! Unknown error detected.
    eeUnknownError,
};

/*! \brief
 * Returns a short string description of an error code.
 *
 * \param[in] errorcode Error code to retrieve the string for.
 * \returns   A constant string corresponding to \p errorcode.
 *
 * If \p errorcode is not one of those defined for ::gmx::ErrorCode,
 * the string corresponding to ::eeUnknownError is returned.
 *
 * This function does not throw.
 */
const char* getErrorCodeString(int errorcode);

/*!\}*/

} // namespace gmx

#endif
