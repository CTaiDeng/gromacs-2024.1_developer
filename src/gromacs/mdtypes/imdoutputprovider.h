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

/*! \libinternal \file
 * \brief
 * Declares gmx::IMDOutputProvider.
 *
 * See \ref page_mdmodules for an overview of this and associated interfaces.
 *
 * \inlibraryapi
 * \ingroup module_mdtypes
 */
#ifndef GMX_MDTYPES_IMDOUTPUTPROVIDER_H
#define GMX_MDTYPES_IMDOUTPUTPROVIDER_H

#include <cstdio>

struct gmx_output_env_t;
struct t_filenm;

namespace gmx
{

/*! \libinternal \brief
 * Interface for handling additional output files during a simulation.
 *
 * This interface provides a mechanism for additional modules to initialize
 * and finalize output files during the simulation.  Writing values to the
 * output files is currently handled elsewhere (e.g., when the module has
 * computed its forces).
 *
 * The interface is not very generic, as it has been written purely based on
 * extraction of existing functions related to electric field handling.
 * Also, the command-line parameters to specify the output files cannot be
 * specified by the module, but are hard-coded in mdrun.
 * This needs to be generalized when more modules are moved to use the
 * interface.
 *
 * \inlibraryapi
 * \ingroup module_mdtypes
 */
class IMDOutputProvider
{
public:
    /*! \brief
     * Initializes file output from a simulation run.
     *
     * \param[in] fplog File pointer for log messages
     * \param[in] nfile Number of files
     * \param[in] fnm   Array of filenames and properties
     * \param[in] bAppendFiles Whether or not we should append to files
     * \param[in] oenv  The output environment for xvg files
     */
    virtual void initOutput(FILE*                   fplog,
                            int                     nfile,
                            const t_filenm          fnm[],
                            bool                    bAppendFiles,
                            const gmx_output_env_t* oenv) = 0;

    //! Finalizes output from a simulation run.
    virtual void finishOutput() = 0;

protected:
    virtual ~IMDOutputProvider() {}
};

} // namespace gmx

#endif
