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
 * Implements helper functions for reading structure factors from datafile
 *
 * \author Alexey Shvetsov <alexxyum@gmail.com>
 * \ingroup module_trajectoryanalysis
 */

#include "scatteringfactors.h"

#include "gromacs/utility/futil.h"
#include "gromacs/utility/strdb.h"

#include "isotope.h"

namespace gmx
{

std::vector<AtomicStructureFactor> readAtomicStructureFactors()
{
    std::vector<AtomicStructureFactor> atomicStructureFactors;
    gmx::FilePtr                       fptr = gmx::openLibraryFile("scatteringfactors.dat");
    char                               line[1000];
    // loop over all non header lines
    while (get_a_line(fptr.get(), line, 1000))
    {
        int    proton;
        char   currentAtomType[8];
        double cohb, cma1, cma2, cma3, cma4, cmb1, cmb2, cmb3, cmb4, cmc;

        if (sscanf(line,
                   "%s %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                   currentAtomType,
                   &proton,
                   &cohb,
                   &cma1,
                   &cma2,
                   &cma3,
                   &cma4,
                   &cmb1,
                   &cmb2,
                   &cmb3,
                   &cmb4,
                   &cmc)
            == 12)
        {
            std::array<double, 4> cma        = { cma1, cma2, cma3, cma4 };
            std::array<double, 4> cmb        = { cmb1, cmb2, cmb3, cmb4 };
            CromerMannParameters  cromerMann = { cma, cmb, cmc };
            AtomicStructureFactor asf        = { currentAtomType, proton, cohb, cromerMann };
            atomicStructureFactors.push_back(asf);
        }
    }
    return atomicStructureFactors;
}


} // namespace gmx
