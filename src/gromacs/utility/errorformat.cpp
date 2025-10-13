/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2010- The GROMACS Authors
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

/*! \internal \file
 * \brief
 * Implements functions declared in errorformat.h.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_utility
 */
#include "gmxpre.h"

#include "errorformat.h"

#include <cctype>
#include <cstdio>
#include <cstring>

#include "gromacs/utility/basenetwork.h"
#include "gromacs/utility/baseversion.h"
#include "gromacs/utility/path.h"
#include "gromacs/utility/programcontext.h"
#include "gromacs/utility/stringutil.h"

namespace gmx
{

/*! \cond internal */
namespace internal
{

void printFatalErrorHeader(FILE* fp, const char* title, const char* func, const char* file, int line)
{
    // In case ProgramInfo is not initialized and there is an issue with the
    // initialization, fall back to "GROMACS".
    const char* programName = "GROMACS";
    try
    {
        programName = getProgramContext().displayName();
    }
    catch (const std::exception&)
    {
    }

    std::fprintf(fp, "\n-------------------------------------------------------\n");
    std::fprintf(fp, "Program:     %s, version %s\n", programName, gmx_version());
    if (file)
    {
        std::fprintf(fp, "Source file: %s (line %d)\n", stripSourcePrefix(file).c_str(), line);
    }
    if (func != nullptr)
    {
        std::fprintf(fp, "Function:    %s\n", func);
    }
    if (gmx_node_num() > 1)
    {
        std::fprintf(fp, "MPI rank:    %d (out of %d)\n", gmx_node_rank(), gmx_node_num());
    }
    std::fprintf(fp, "\n");
    std::fprintf(fp, "%s:\n", title);
}

void printFatalErrorMessageLine(FILE* fp, const char* text, int indent)
{
    gmx::TextLineWrapper wrapper;
    wrapper.settings().setLineLength(78 - indent);
    size_t lineStart = 0;
    size_t length    = std::strlen(text);
    while (lineStart < length)
    {
        size_t nextLineStart = wrapper.findNextLine(text, lineStart);
        int    lineLength    = static_cast<int>(nextLineStart - lineStart);
        while (lineLength > 0 && std::isspace(text[lineStart + lineLength - 1]))
        {
            --lineLength;
        }
        std::fprintf(fp, "%*s%.*s\n", indent, "", lineLength, text + lineStart);
        lineStart = nextLineStart;
    }
}

void printFatalErrorFooter(FILE* fp)
{
    std::fprintf(fp, "\n");
    std::fprintf(fp,
                 "For more information and tips for troubleshooting, please check the GROMACS\n"
                 "website at https://manual.gromacs.org/current/user-guide/run-time-errors.html");
    std::fprintf(fp, "\n-------------------------------------------------------\n");
}

} // namespace internal
//! \endcond

} // namespace gmx
