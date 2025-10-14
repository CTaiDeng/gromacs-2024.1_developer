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

/*! \file
 * \brief
 * Declares gmx::IOutputAdapter interface for modifying coordinate
 * file structures before writing them to disk.
 *
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 * \inlibraryapi
 * \ingroup module_coordinateio
 */
#ifndef GMX_COORDINATEIO_IOUTPUTADAPTER_H
#define GMX_COORDINATEIO_IOUTPUTADAPTER_H

#include <memory>

#include "gromacs/utility/classhelpers.h"

#include "coordinatefileenums.h"

struct t_trxframe;

namespace gmx
{

/*!\brief
 * OutputAdapter class for handling trajectory file flag setting and processing.
 *
 * This interface provides the base point upon which modules that modify trajectory frame
 * datastructures should be build. The interface itself does not provide any direct means
 * to modify the data, but only gives the virtual method to perform work on a
 * t_trxframe object. Classes that modify trajectory frames should implement this interface.
 *
 * \inlibraryapi
 * \ingroup module_coordinateio
 *
 */
class IOutputAdapter
{
public:
    /*! \brief
     * Default constructor for IOutputAdapter interface.
     */
    IOutputAdapter() {}
    virtual ~IOutputAdapter() {}
    //! Move constructor for old object.
    explicit IOutputAdapter(IOutputAdapter&& old) noexcept = default;

    /*! \brief
     * Change t_trxframe according to user input.
     *
     * \param[in] framenumber Frame number as reported from the
     *                        trajectoryanalysis framework or set by user.
     * \param[in,out] input   Pointer to trajectory analysis frame that will
     *                        be worked on.
     */
    virtual void processFrame(int framenumber, t_trxframe* input) = 0;

    /*! \brief
     * Checks that the abilities of the output writer are sufficient for this adapter.
     *
     * It can happen that a method to write coordinate files does not match with
     * a requested operation on the input data (e.g. the user requires velocities or
     * forces to be written to a PDB file).
     * To check those dependencies, derived classes need to implement a version of this
     * function to make sure that only matching methods can be used.
     *
     * \param[in] abilities The abilities of an output method that need to be checked against
     *                      the dependencies created by using the derived method.
     * \throws InconsistentInputError If dependencies can not be matched to abilities.
     */
    virtual void checkAbilityDependencies(unsigned long abilities) const = 0;

    GMX_DISALLOW_COPY_AND_ASSIGN(IOutputAdapter);
};

//! Smart pointer to manage the frame adapter object.
using OutputAdapterPointer = std::unique_ptr<IOutputAdapter>;

} // namespace gmx

#endif
