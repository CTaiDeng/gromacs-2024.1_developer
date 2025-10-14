/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2023- The GROMACS Authors
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

/*! \internal \file
 * \brief
 * Declares base class for Debye Scattering
 *
 * \author Alexey Shvetsov <alexxyum@gmail.com>
 * \ingroup module_trajectoryanalysis
 */

#ifndef GMX_TRAJECTORYANALYSIS_SCATTERING_DEBYE_H
#define GMX_TRAJECTORYANALYSIS_SCATTERING_DEBYE_H

#include <string>
#include <unordered_map>
#include <vector>

#include "gromacs/math/vec.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/selection/selection.h"
#include "gromacs/topology/topology.h"

#include "isotope.h"
#include "scatteringfactors.h"

namespace gmx
{

/*! \internal \brief
 * structure to store pair dist units
 */
struct PairDistValue
{
    //! index of atom i
    std::size_t atomI;
    //! index of atom j
    std::size_t atomJ;
    //! distance between atoms i and j
    float distanceIJ;
};

/*! \internal \brief
 * Base class for computing SANS and SAXS using Debye Method
 *
 * \c ComputeDebyeScattering uses Debye formula for scattering:
 * \f[I(s) = \sum_{i} \sum_{j} f_i(s) * f_j(s) * \frac{sin(s*r_{ij})}{s*r_{ij}}\f]
 * where \f[ r_{ij} = \left| \vec{r_i} - \vec{r_j} \right| \f] between atoms i and j
 * and \f[f_i(s)\f], \f[f_j(s)\f] are atomic structure factors for atoms i and j.
 */
class ComputeDebyeScattering
{
public:
    virtual ~ComputeDebyeScattering() = default;

    //! retrieves scattering length based on atom index
    virtual double getScatteringLength(int i, double q) = 0;

    //! Compute Pair distances for atoms using Direct Method
    void computeDirectPairDistancesHistogram(t_pbc* pbc, Selection sel);

    //! Compute Pair distances for atoms using MonteCarlo Method
    void computeMonteCarloPairDistancesHistogram(t_pbc* pbc, Selection sel, float coverage, int seed);

    //! computes the scattering intensity for a given scattering angle q
    void computeIntensity();

    //! computes the scattering intensity for a given scattering zerro angle
    double computeIntensityZeroQ();

    double getIntensity(size_t qi);

    //! computes atomic form factors
    double getFormFactor(int i, int j, double q);

    //! Add list of Q values
    void addQList(std::vector<double> qList);

    //! Set binwidht for histogramm
    void setBinWidth(double binWidth);

    //! Set max distance for histogram from box size
    void getMaxDist(matrix box);

    //! Initialize pair distance histogram
    void initPairDistHist();

    //! Clear histogram
    void clearHist();

private:
    //! binwidth for P(r) hist
    double binWidth_ = 0.F;
    //! maximum possible distance in a box
    double maxDist_ = 0.F;
    //! maximum number of bins in hist
    std::size_t maxHIndex_ = 0;
    //! List of Q values to calculate scattering
    std::vector<double> qValues_;
    //! List of distance in hist
    std::vector<double> histRValues_;
    //! Intensity of scattering
    std::vector<double> intensity_;
    //! List of sf*distance values in hist in case of SANS when SF dont depend on Q
    std::vector<double> sfDistValues_;
    //! List of sf*distance values in hist in case of SAXS when SF depend on Q
    std::vector<std::vector<double>> sfQDependDistValues_;
    //! add atom pair for histogram
    void addPairToHist(PairDistValue pair);

protected:
    //! set if structure factor depend on Q value (e.g. for SAXS)
    bool sfDepenOnQ_ = false;
};

} // namespace gmx

#endif // GMX_TRAJECTORYANALYSIS_SCATTERING_DEBYE_H
