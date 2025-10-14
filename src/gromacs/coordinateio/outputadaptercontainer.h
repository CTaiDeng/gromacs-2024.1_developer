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
 * Declares gmx::OutputAdapterContainer, a storage object for
 * multiple outputadapters derived from the IOutputadaper interface.
 *
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 * \inlibraryapi
 * \ingroup module_coordinateio
 */
#ifndef GMX_COORDINATEIO_OUTPUTADAPTERCONTAINER_H
#define GMX_COORDINATEIO_OUTPUTADAPTERCONTAINER_H

#include <memory>
#include <vector>

#include "gromacs/coordinateio/ioutputadapter.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/enumerationhelpers.h"

#include "coordinatefileenums.h"

namespace gmx
{

/*! \libinternal \brief
 * Storage for output adapters that modify the state of a t_trxframe object.
 *
 * The OutputAdapterContainer is responsible for storing the number of
 * OutputAdapters, as well as the bitmask representing the current requirements
 * for constructing an CoordinateFile object with the modules registered. It is responsible
 * for ensuring that no module can be registered multiple times, and that the
 * correct order for some modifications is observed (e.g. we can not reduce the
 * number of coordinates written to a file before we have set all the other flags).
 * Any other behaviour indicates a programming error and triggers an assertion.
 *
 * The abilities that need to be cross checked for the container are usually constrained
 * by the file format the coordinate data will be written to. When declaring new abilities,
 * these must match the file type for the output.
 *
 * \todo Keeping track of those abilities has to be the responsibility of an object
 *       implementing and interface that declares it capabilities and will execute the
 *       the function of writing to a file.
 * \todo This could be changed to instead construct the container with a pointer to an
 *       ICoordinateOutputWriter that can be passed to the IOutputAdapter modules to check
 *       their cross-dependencies.
 */
class OutputAdapterContainer
{
public:
    //! Only allow constructing the container with defined output abilities.
    explicit OutputAdapterContainer(unsigned long abilities) : abilities_(abilities) {}
    //! Allow abilities to be also defined using the enum class.
    explicit OutputAdapterContainer(CoordinateFileFlags abilities) :
        abilities_(convertFlag(abilities))
    {
    }

    /*! \brief
     * Add an adapter of a type not previously added.
     *
     * Only one adapter of each type can be registered, and the order of adapters
     * is predefined in the underlying storage object.
     * Calls internal checks to make sure that the new adapter does not violate
     * any of the preconditions set to make an CoordinateFile object containing
     * the registered modules.
     *
     * \param[in] adapter unique_ptr to adapter, with container taking ownership here.
     * \param[in] type What kind of adapter is being added.
     * \throws InternalError When registering an adapter of a type already registered .
     * \throws InconsistentInputError When incompatible modules are added.
     */
    void addAdapter(OutputAdapterPointer adapter, CoordinateFileFlags type);

    //! Get vector of all registered adapters.
    ArrayRef<const OutputAdapterPointer> getAdapters() { return outputAdapters_; }
    //! Get info if we have any registered adapters.
    bool isEmpty() const;

private:
    //! Array of registered modules.
    EnumerationArray<CoordinateFileFlags, OutputAdapterPointer> outputAdapters_;
    //! Construction time bitmask declaring what the OutputManager can do.
    unsigned long abilities_ = convertFlag(CoordinateFileFlags::Base);
};

} // namespace gmx

#endif
