/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 1991- The GROMACS Authors
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

#ifndef GMX_FILEIO_XTCIO_H
#define GMX_FILEIO_XTCIO_H

#include <filesystem>

#include "gromacs/math/vectypes.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/real.h"

struct t_fileio;

/* All functions return 1 if successful, 0 otherwise
 * bOK tells if a frame is not corrupted
 */

/* Note that XTC was implemented to use xdr_int for the step number,
 * which is defined by the standard to be signed and 32 bit. We didn't
 * design the format to be extensible, so we can't fix the fact that
 * after 2^31 frames, step numbers will wrap to be
 * negative. Fortunately, this tends not to cause serious problems,
 * and we've fixed it in TNG. */

struct t_fileio* open_xtc(const std::filesystem::path& filename, const char* mode);
/* Open a file for xdr I/O */

void close_xtc(struct t_fileio* fio);
/* Close the file for xdr I/O */

int read_first_xtc(struct t_fileio* fio,
                   int*             natoms,
                   int64_t*         step,
                   real*            time,
                   matrix           box,
                   rvec**           x,
                   real*            prec,
                   gmx_bool*        bOK);
/* Open xtc file, read xtc file first time, allocate memory for x */

int read_next_xtc(struct t_fileio* fio, int natoms, int64_t* step, real* time, matrix box, rvec* x, real* prec, gmx_bool* bOK);
/* Read subsequent frames */

int write_xtc(struct t_fileio* fio, int natoms, int64_t step, real time, const rvec* box, const rvec* x, real prec);
/* Write a frame to xtc file */

#endif
