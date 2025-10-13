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

/*! \file
 * \brief
 * Provides function to open output files automatically (with some X11
 * programs).
 *
 * \inpublicapi
 * \ingroup module_commandline
 */
#ifndef GMX_COMMANDLINE_VIEWIT_H
#define GMX_COMMANDLINE_VIEWIT_H

struct gmx_output_env_t;
struct t_filenm;

/*! \brief
 * Executes an external (X11) command to view a file.
 *
 * Currently eps, xpm, xvg and pdb are supported.
 * Default programs are provided, can be overriden with environment vars
 * (but note that if the caller provides program-specific \p opts, setting the
 * environment variable most likely breaks things).
 */
void do_view(const gmx_output_env_t* oenv, const char* fn, const char* opts);

/*! \brief
 * Calls do_view() for all viewable output files.
 */
void view_all(const gmx_output_env_t* oenv, int nf, t_filenm fnm[]);

#endif
