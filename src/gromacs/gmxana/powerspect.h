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

#ifndef GMX_GMXANA_POWERSPECT_H
#define GMX_GMXANA_POWERSPECT_H

#include <string>
#include <vector>

#include "gromacs/gmxana/interf.h"
#include "gromacs/utility/real.h"

namespace gmx
{
template<typename>
class ArrayRef;
}

extern void powerspectavg(real*** interface, int t, int xbins, int ybins, gmx::ArrayRef<const std::string> outfiles);

extern void powerspectavg_intf(t_interf***                      if1,
                               t_interf***                      if2,
                               int                              t,
                               int                              xbins,
                               int                              ybins,
                               gmx::ArrayRef<const std::string> outfiles);

#endif
