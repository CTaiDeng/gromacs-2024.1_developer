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
 * Declares gmx::SetForces.
 *
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 * \inpublicapi
 * \ingroup module_coordinateio
 */
#ifndef GMX_COORDINATEIO_SETFORCES_H
#define GMX_COORDINATEIO_SETFORCES_H

#include <memory>

#include "gromacs/coordinateio/coordinatefileenums.h"
#include "gromacs/coordinateio/ioutputadapter.h"

namespace gmx
{

/*!\brief
 * SetForces class allows changing writing of forces to file.
 *
 * This class allows the user to define if forces should be written
 * to the output coordinate file, and checks if they are available from the
 * currently processed data.
 *
 * \inpublicapi
 * \ingroup module_coordinateio
 *
 */
class SetForces : public IOutputAdapter
{
public:
    /*! \brief
     * Construct SetForces object with choice for boolean value.
     *
     * Can be used to initialize SetForces from outside of trajectoryanalysis
     * with the user specified option to write coordinate forces or not.
     */
    explicit SetForces(ChangeSettingType force) : force_(force)
    {
        if (force == ChangeSettingType::Never)
        {
            moduleRequirements_ = CoordinateFileFlags::Base;
        }
        else
        {
            moduleRequirements_ = CoordinateFileFlags::RequireForceOutput;
        }
    }
    /*! \brief
     *  Move constructor for SetForces.
     */
    SetForces(SetForces&& old) noexcept = default;

    ~SetForces() override {}

    /*! \brief
     * Change coordinate frame information for output.
     *
     * In this case, the correct flag for writing the forces is applied
     * to the output frame, depending on user selection and availability
     * in the input data.
     *
     * \param[in] input Coordinate frame to be modified later.
     */
    void processFrame(int /*framenumner*/, t_trxframe* input) override;

    void checkAbilityDependencies(unsigned long abilities) const override;

private:
    /*! \brief
     * Flag to specify if forces should be written.
     *
     * Internal storage for the user choice for writing coordinate forces.
     */
    ChangeSettingType force_;
    //! Local requirements to be determined from user input.
    CoordinateFileFlags moduleRequirements_;
};

//! Smart pointer to manage the object.
using SetForcesPointer = std::unique_ptr<SetForces>;

} // namespace gmx

#endif
