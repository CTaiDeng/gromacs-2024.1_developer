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

/*! \file
 * \brief
 * Declares gmx::SetPrecision.
 *
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 * \inpublicapi
 * \ingroup module_coordinateio
 */
#ifndef GMX_FILEIO_SETPRECISION_H
#define GMX_FILEIO_SETPRECISION_H

#include <algorithm>

#include "gromacs/coordinateio/coordinatefileenums.h"
#include "gromacs/coordinateio/ioutputadapter.h"
#include "gromacs/utility/real.h"

namespace gmx
{

/*!\brief
 * SetPrecision class allows changing file writing precision.
 *
 * This class allows the user to define the precision for writing
 * coordinate data to output files.
 *
 * \inpublicapi
 * \ingroup module_coordinateio
 *
 */
class SetPrecision : public IOutputAdapter
{
public:
    /*! \brief
     * Construct SetPrecision object with user defined value.
     *
     * Can be used to initialize SetPrecision from outside of trajectoryanalysis
     * with the user specified option to change precision or not.
     *
     * \param[in] precision User defined value for output precision in file types that support it.
     */
    explicit SetPrecision(int precision) : precision_(precision)
    {
        // Only request special treatment if precision is not the default.
        if (precision == 3)
        {
            moduleRequirements_ = CoordinateFileFlags::Base;
        }
        else
        {
            moduleRequirements_ = CoordinateFileFlags::RequireChangedOutputPrecision;
        }
    }
    /*! \brief
     *  Move constructor for SetPrecision.
     */
    SetPrecision(SetPrecision&& old) noexcept = default;

    ~SetPrecision() override {}

    void processFrame(int /*framenumber*/, t_trxframe* input) override;

    void checkAbilityDependencies(unsigned long abilities) const override;

private:
    //! User specified changes to default precision.
    int precision_;
    //! Module requirements dependent on user input.
    CoordinateFileFlags moduleRequirements_;
};

//! Smart pointer to manage the outputselector object.
using SetPrecisionPointer = std::unique_ptr<SetPrecision>;

} // namespace gmx

#endif
