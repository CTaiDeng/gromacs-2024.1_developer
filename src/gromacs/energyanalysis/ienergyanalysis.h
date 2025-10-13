/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2019- The GROMACS Authors
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
 * Declares gmx::IEnergyAnalysis
 *
 * \author David van der Spoel <david.vanderspoel@icm.uu.se>
 * \ingroup module_energyanalysis
 */
#ifndef GMX_ENERGYANALYSIS_IENERGYANALYSIS_H
#define GMX_ENERGYANALYSIS_IENERGYANALYSIS_H

#include <string>
#include <vector>

#include "gromacs/commandline/cmdlineoptionsmodule.h"
#include "gromacs/options.h"

struct gmx_output_env_t;
struct t_enxframe;

namespace gmx
{

/*! \libinternal \brief
 * Convenience structure for keeping energy name and unit together
 */
struct EnergyNameUnit
{
    //! Name of this energy term
    std::string energyName;
    //! Unit of this energy term
    std::string energyUnit;
};

/*! \libinternal
 * \brief
 * Interface class overloaded by the separate energy modules.
 */
class IEnergyAnalysis
{
public:
    //! Obligatory virtual destructor
    virtual ~IEnergyAnalysis() {}

    //! Initiate the command line options
    virtual void initOptions(IOptionsContainer* options, ICommandLineOptionsModuleSettings* settings) = 0;

    /*! \brief
     * Does the initiation of the analysis of the file
     * \param[in] energyNamesAndUnits   Names and units of the energy terms.
     * \param[in] oenv  GROMACS output environment
     */
    virtual void initAnalysis(ArrayRef<const EnergyNameUnit> energyNamesAndUnits,
                              const gmx_output_env_t&        oenv) = 0;

    /*! \brief
     * Analyse one frame and stores the results in memory
     * \param[in] fr The energy data frame
     * \param[in] oenv  GROMACS output environment. This is needed in some cases
     *                  where the output to be written depends on the content of
     *                  the energy file.
     */
    virtual void analyzeFrame(t_enxframe* fr, const gmx_output_env_t& oenv) = 0;

    /*! \brief
     * Finalize reading and write output files.
     * \param[in] oenv  GROMACS output environment.
     */
    virtual void finalizeAnalysis(const gmx_output_env_t& oenv) = 0;

    /*! \brief
     * View the output file(s)
     * \param[in] oenv  GROMACS output environment.
     */
    virtual void viewOutput(const gmx_output_env_t& oenv) = 0;
};

//! Pointer to the EnergyAnalysisModule classes.
using IEnergyAnalysisPointer = std::unique_ptr<IEnergyAnalysis>;

} // namespace gmx

#endif
