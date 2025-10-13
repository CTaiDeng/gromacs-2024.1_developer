/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2023- The GROMACS Authors
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
 * Implements class for SAXS Debye Scattering
 *
 * \author Alexey Shvetsov <alexxyum@gmail.com>
 * \ingroup module_trajectoryanalysis
 */

#include "gmxpre.h"

#include "scattering-debye-saxs.h"

#include <cmath>

#include <vector>

#include "gromacs/math/units.h"

namespace gmx
{


SaxsDebye::SaxsDebye(std::vector<Isotope> isotopes, const std::vector<double>& qList) :
    isotopes_(std::move(isotopes))
{
    sfDepenOnQ_                                             = true;
    std::vector<AtomicStructureFactor> atomicScatterFactors = readAtomicStructureFactors();
    for (auto q : qList)
    {
        for (auto scatter : atomicScatterFactors)
        {

            double scattering = scatter.xrayCromerMannParameters.c;
            double q4pi       = q / (4.0 * M_PI);
            for (int j = 0; j < 4; j++)
            {
                scattering += scatter.xrayCromerMannParameters.a[j]
                              * exp(-scatter.xrayCromerMannParameters.b[j] * q4pi * q4pi);
            }
            Isotope isotope       = getIsotopeFromString(scatter.isotope);
            auto    pair          = std::make_pair(static_cast<int>(isotope), q);
            scatterFactors_[pair] = scattering;
        }
    }
}

double SaxsDebye::getScatteringLength(int i, double q)
{
    auto   pair       = std::make_pair(static_cast<int>(isotopes_[i]), q);
    double scattering = scatterFactors_[pair];
    return scattering;
}


} // namespace gmx
