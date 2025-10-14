/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2019- The GROMACS Authors
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

/*! \libinternal \file
 * \brief
 * Declares gmx::EnergyAnalysisFrame
 *
 * \author David van der Spoel <david.vanderspoel@icm.uu.se>
 * \ingroup module_energyanalysis
 */
#ifndef GMX_ENERGYANALYSIS_ENERGYANALYSISFRAME_H
#define GMX_ENERGYANALYSIS_ENERGYANALYSISFRAME_H

#include "gromacs/utility/basedefinitions.h"

namespace gmx
{

/*! \libinternal
 * \brief
 * Class describing an energy frame, that is the all the data
 * stored for one energy term at one time step in an energy file.
 * \todo Make this supersede t_enxframe.
 */
class EnergyAnalysisFrame
{
public:
    /*! \brief
     * Constructor
     * \param[in] time The time
     * \param[in] step The MD step
     * \param[in] energyAtTime The instantaneous energy
     * \param[in] numSteps The number of statistics points
     * \param[in] energySumOverNumSteps The sum of energies over previous steps
     * \param[in] energyVarianceOverNumSteps The variance over previous steps
     */
    EnergyAnalysisFrame(double  time,
                        int64_t step,
                        double  energyAtTime,
                        int     numSteps,
                        double  energySumOverNumSteps,
                        double  energyVarianceOverNumSteps) :
        time_(time),
        step_(step),
        energyAtTime_(energyAtTime),
        numSteps_(numSteps),
        energySumOverNumSteps_(energySumOverNumSteps),
        energyVarianceOverNumSteps_(energyVarianceOverNumSteps)
    {
    }
    /*! \brief Return the time
     *
     * \return The time
     */
    double time() const { return time_; }
    /*! \brief Return the step
     *
     * \return the step
     */
    int64_t step() const { return step_; }
    /*! \brief Return the instantaneous energy
     *
     * \return the instantaneous energy
     */
    double energyAtTime() const { return energyAtTime_; }
    /*! \brief Return the number of statistics points
     *
     * \return the number of statistics points
     */
    int numSteps() const { return numSteps_; }
    /*! \brief Return the sum of energies
     *
     * \return the sum of energies
     */
    double energySumOverNumSteps() const { return energySumOverNumSteps_; }
    /*! \brief Return the variance of energies
     *
     * \return the variance of energies
     */
    double energyVarianceOverNumSteps() const { return energyVarianceOverNumSteps_; }

private:
    //! The time in the simulation
    double time_;
    //! The step in the simulation
    int64_t step_;
    //! The energy at this time point
    double energyAtTime_;
    //! The number of statistics points
    int numSteps_;
    //! The sum of energies over nsum_ previous steps
    double energySumOverNumSteps_;
    //! The variance over nsum_ previous steps
    double energyVarianceOverNumSteps_;
};

} // namespace gmx
#endif
