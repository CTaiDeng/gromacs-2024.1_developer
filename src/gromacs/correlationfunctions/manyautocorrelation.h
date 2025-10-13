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

/*! \libinternal
 * \file
 * \brief
 * Declares routine for computing many correlation functions using OpenMP
 *
 * \author David van der Spoel <david.vanderspoel@icm.uu.se>
 * \inlibraryapi
 * \ingroup module_correlationfunctions
 */
#ifndef GMX_MANYAUTOCORRELATION_H
#define GMX_MANYAUTOCORRELATION_H

#include <vector>

#include "gromacs/fft/fft.h"
#include "gromacs/utility/real.h"

/*! \brief
 * Perform many autocorrelation calculations.
 *
 * This routine performs many autocorrelation function calculations using FFTs.
 * The GROMACS FFT library wrapper is employed. On return the c vector contain
 * a symmetric function that is useful for further FFT:ing, for instance in order to
 * compute spectra.
 *
 * The vectors c[i] should all have the same length, but this is not checked for.
 *
 * The c arrays will be extend and filled with zero beyond ndata before
 * computing the correlation.
 *
 * The functions uses OpenMP parallellization.
 *
 * \param[inout] c Data array
 * \return fft error code, or zero if everything went fine (see fft/fft.h)
 * \throws gmx::InconsistentInputError if the input is inconsistent.
 */
int many_auto_correl(std::vector<std::vector<real>>* c);

#endif
