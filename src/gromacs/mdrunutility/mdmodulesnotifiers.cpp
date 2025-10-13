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
 *
 * \brief Implements routines in mdmodulesnotifiers.h
 *
 * \author Hubert Santuz <hubert.santuz@gmail.com>
 * \ingroup module_mdrunutility
 */
#include "gmxpre.h"

#include "mdmodulesnotifiers.h"

#include "gromacs/mdtypes/inputrec.h"


namespace gmx
{


EnergyCalculationFrequencyErrors::EnergyCalculationFrequencyErrors(int64_t energyCalculationIntervalInSteps) :
    energyCalculationIntervalInSteps_(energyCalculationIntervalInSteps)
{
}

std::int64_t EnergyCalculationFrequencyErrors::energyCalculationIntervalInSteps() const
{
    return energyCalculationIntervalInSteps_;
}

void EnergyCalculationFrequencyErrors::addError(const std::string& errorMessage)
{
    errorMessages_.push_back(errorMessage);
}

const std::vector<std::string>& EnergyCalculationFrequencyErrors::errorMessages() const
{
    return errorMessages_;
}


EnsembleTemperature::EnsembleTemperature(const t_inputrec& ir)
{
    if (haveConstantEnsembleTemperature(ir))
    {
        constantEnsembleTemperature_ = std::make_optional(constantEnsembleTemperature(ir));
    }
    else
    {
        constantEnsembleTemperature_ = std::nullopt;
    }
}

} // namespace gmx
