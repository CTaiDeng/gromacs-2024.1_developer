/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2018- The GROMACS Authors
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

/*! \internal
 * \brief Defines the dispatch function for the .mdp integrator field.
 *
 * \author David van der Spoel <david.vanderspoel@icm.uu.se>
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \ingroup module_mdrun
 */
#include "gmxpre.h"

#include "legacysimulator.h"

#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/utility/exceptions.h"

namespace gmx
{
//! \brief Run the correct integrator function.
void LegacySimulator::run()
{
    switch (inputRec_->eI)
    {
        case IntegrationAlgorithm::MD:
        case IntegrationAlgorithm::BD:
        case IntegrationAlgorithm::SD1:
        case IntegrationAlgorithm::VV:
        case IntegrationAlgorithm::VVAK:
            if (!EI_DYNAMICS(inputRec_->eI))
            {
                GMX_THROW(APIError(
                        "do_md integrator would be called for a non-dynamical integrator"));
            }
            if (doRerun_)
            {
                do_rerun();
            }
            else
            {
                do_md();
            }
            break;
        case IntegrationAlgorithm::Mimic:
            if (doRerun_)
            {
                do_rerun();
            }
            else
            {
                do_mimic();
            }
            break;
        case IntegrationAlgorithm::Steep: do_steep(); break;
        case IntegrationAlgorithm::CG: do_cg(); break;
        case IntegrationAlgorithm::NM: do_nm(); break;
        case IntegrationAlgorithm::LBFGS: do_lbfgs(); break;
        case IntegrationAlgorithm::TPI:
        case IntegrationAlgorithm::TPIC:
            if (!EI_TPI(inputRec_->eI))
            {
                GMX_THROW(APIError("do_tpi integrator would be called for a non-TPI integrator"));
            }
            do_tpi();
            break;
        case IntegrationAlgorithm::SD2Removed:
            GMX_THROW(NotImplementedError("SD2 integrator has been removed"));
        default: GMX_THROW(APIError("Non existing integrator selected"));
    }
}

} // namespace gmx
