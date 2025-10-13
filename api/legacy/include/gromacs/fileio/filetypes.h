/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 1991- The GROMACS Authors
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

#ifndef GMX_FILEIO_FILETYPES_H
#define GMX_FILEIO_FILETYPES_H

#include <filesystem>

#include "gromacs/utility/basedefinitions.h"

/* this enum should correspond to the array deffile in filetypes.cpp */
enum GromacsFileType
{
    efMDP,
    efTRX,
    efTRO,
    efTRN,
    efTRR,
    efCOMPRESSED,
    efXTC,
    efTNG,
    efEDR,
    efSTX,
    efSTO,
    efGRO,
    efG96,
    efPDB,
    efBRK,
    efENT,
    efESP,
    efPQR,
    efCPT,
    efLOG,
    efXVG,
    efOUT,
    efNDX,
    efTOP,
    efITP,
    efTPS,
    efTPR,
    efTEX,
    efRTP,
    efATP,
    efHDB,
    efDAT,
    efDLG,
    efMAP,
    efEPS,
    efMAT,
    efM2P,
    efMTX,
    efEDI,
    efCUB,
    efXPM,
    efRND,
    efCSV,
    efQMI,
    efNR
};

const char* ftp2ext(int ftp);
/* Return extension for filetype */

const char* ftp2ext_generic(int ftp);
/* Return extension for filetype, and a generic name for generic types
   (e.g. trx)*/

const char* ftp2ext_with_dot(int ftp);
/* Return extension for filetype with a leading dot */

int ftp2generic_count(int ftp);
/* Return the number of filetypes for a generic filetype */

const int* ftp2generic_list(int ftp);
/* Return the list of filetypes for a generic filetype */

const char* ftp2desc(int ftp);
/* Return description for file type */

const char* ftp2defnm(int ftp);
/* Return default file name for file type */

const char* ftp2defopt(int ftp);
/* Return default option name for file type */

gmx_bool ftp_is_text(int ftp);
gmx_bool ftp_is_xdr(int ftp);

//! Return the filetype corresponding to filename
int fn2ftp(const std::filesystem::path& fn);

//! Return the filetype corresponding to filename. Overload to handle nullptr.
int fn2ftp(const char* path);

#endif
