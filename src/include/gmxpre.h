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
 * Prerequisite header file for \Gromacs build.
 *
 * This header should be included as the first header in all source files, but
 * not in header files.  It is intended to contain definitions that must appear
 * before any other code to work properly (e.g., macro definitions that
 * influence behavior of system headers).  This frees other code from include
 * order dependencies that may raise from requirements of getting these
 * definitions from some header.
 *
 * The definitions here should be kept to a minimum, and should be as static as
 * possible (typically not change as a result of user choices in the build
 * system), as any change will trigger a full rebuild.  Avoid including any
 * actual headers to not hide problems with include-what-you-use, and to keep
 * build times to a minimum.  Also, installer headers should avoid relying on
 * the definitions from here (if possible), as this header will not be
 * available to the user.
 *
 * \inlibraryapi
 */
//! \cond
#ifdef HAVE_CONFIG_H
#    include "gmxpre-config.h"
#endif

/* We use a few GNU functions for thread affinity and other low-level stuff.
 * However, all such uses should be accompanied by #ifdefs and a feature test
 * at CMake level, so that the actual uses will be compiled only when available.
 * But since the define affects system headers, it should be defined before
 * including any system headers, and this is a robust location to do that.
 * If this were defined only in source files that needed it, it would clutter
 * the list of includes somewhere close to the beginning and make automatic
 * sorting of the includes more difficult.
 */
#ifndef _GNU_SOURCE
#    define _GNU_SOURCE 1
#endif

#if GMX_FAHCORE
#    define FULLINDIRECT 1
#    define USE_FAH_XDR 1
#    include "swindirect.h"
#endif
//! \endcond
