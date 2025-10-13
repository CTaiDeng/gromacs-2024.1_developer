/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2010- The GROMACS Authors
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
 * Defines flags used in option implementation.
 *
 * Symbols in this header are considered an implementation detail, and should
 * not be accessed outside the module.
 * Because of details in the implementation, it is still installed.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_options
 */
#ifndef GMX_OPTIONS_OPTIONFLAGS_H
#define GMX_OPTIONS_OPTIONFLAGS_H

#include "gromacs/utility/flags.h"

namespace gmx
{

/*! \cond libapi */
/*! \libinternal \brief
 * Flags for options.
 *
 * These flags are not part of the public interface, even though they are in an
 * installed header.  They are needed in a few template class implementations.
 *
 * \todo
 * The flags related to default values are confusing, consider reorganizing
 * them.
 */
enum OptionFlag : uint64_t
{
    //! %Option has been set.
    efOption_Set = 1 << 0,
    //! The current value of the option is a programmatic default value.
    efOption_HasDefaultValue = 1 << 1,
    //! An explicit default value has been provided for the option.
    efOption_ExplicitDefaultValue = 1 << 2,
    /*! \brief
     * Next assignment to the option clears old values.
     *
     * This flag is set when a new option source starts, such that values
     * from the new source will overwrite old ones.
     */
    efOption_ClearOnNextSet = 1 << 3,
    //! %Option is required to be set.
    efOption_Required = 1 << 4,
    //! %Option can be specified multiple times.
    efOption_MultipleTimes = 1 << 5,
    //! %Option is hidden from standard help.
    efOption_Hidden = 1 << 6,
    /*! \brief
     * %Option value is a vector, but a single value is also accepted.
     *
     * \see AbstractOption::setVector()
     */
    efOption_Vector = 1 << 8,
    //! %Option has a defaultValueIfSet() specified.
    efOption_DefaultValueIfSetExists = 1 << 11,
    //! %Option does not support default values.
    efOption_NoDefaultValue = 1 << 9,
    /*! \brief
     * Storage object does its custom checking for minimum value count.
     *
     * If this flag is set, the class derived from OptionStorageTemplate should
     * implement processSetValues(), processAll(), and possible other functions
     * it provides such that it always fails if not enough values are provided.
     * This is useful to override the default check, which is done in
     * OptionStorageTemplate::processSet().
     */
    efOption_DontCheckMinimumCount = 1 << 10
};

//! \libinternal Holds a combination of ::OptionFlag values.
typedef FlagsTemplate<OptionFlag> OptionFlags;
//! \endcond

} // namespace gmx

#endif
