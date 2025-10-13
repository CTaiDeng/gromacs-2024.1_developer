/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2018- The GROMACS Authors
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

#ifndef GMX_TOOLS_REPORT_METHODS_H
#define GMX_TOOLS_REPORT_METHODS_H

#include <string>

#include "gromacs/commandline/cmdlineoptionsmodule.h"
#include "gromacs/utility/filestream.h"
#include "gromacs/utility/textwriter.h"

struct gmx_mtop_t;
struct t_inputrec;

namespace gmx
{

class ReportMethodsInfo
{
public:
    static LIBGROMACS_EXPORT const char     name[];
    static LIBGROMACS_EXPORT const char     shortDescription[];
    static ICommandLineOptionsModulePointer create();
};

// Helper functions of the class

/*! \brief
 * Write appropiate Header to output stream.
 *
 * \param[in] writer TextWriter object for writing information.
 * \param[in] text String with the header before writing.
 * \param[in] section String with section text for header.
 * \param[in] writeFormattedText If we need to format the text for LaTeX output or not
 */
void writeHeader(TextWriter* writer, const std::string& text, const std::string& section, bool writeFormattedText);

/*! \brief
 * Write information about the molecules in the system.
 *
 * This method should write all possible information about
 * the molecular composition of the system.
 *
 * \param[in] writer TextWriter object for writing information.
 * \param[in] top Local topology used to derive the information to write out.
 * \param[in] writeFormattedText Decide if we want formatted text output or not.
 *
 */
void writeSystemInformation(TextWriter* writer, const gmx_mtop_t& top, bool writeFormattedText);

/*! \brief
 * Write information about system parameters.
 *
 * This method writes the basic information for the system parameters
 * and simulation settings as reported in the \p ir.
 *
 * \param[in] writer TextWriter object for writing information.
 * \param[in] ir Reference to inputrec of the run input.
 * \param[in] writeFormattedText Decide if we want formatted text output or not.
 */
void writeParameterInformation(TextWriter* writer, const t_inputrec& ir, bool writeFormattedText);

/*! \brief
 * Wrapper for writing out information.
 *
 * This function is actual called from within the run method
 * to write the information to the terminal or to file.
 * New write out methods should be added to it instead of adding them in run.
 *
 * \param[in] outputStream The filestream used to write the information to.
 * \param[in] ir Reference to inputrec of the run input.
 * \param[in] top Local topology used to derive the information to write out.
 * \param[in] writeFormattedText Decide if we want formatted text output or not.
 * \param[in] notStdout Bool to see if we can close the file after writing or not in case of stdout.
 */
void writeInformation(TextOutputFile*   outputStream,
                      const t_inputrec& ir,
                      const gmx_mtop_t& top,
                      bool              writeFormattedText,
                      bool              notStdout);

} // namespace gmx

#endif
