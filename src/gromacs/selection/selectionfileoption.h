/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2012- The GROMACS Authors
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
 * Declares gmx::SelectionFileOption and gmx::SelectionFileOptionInfo.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \inlibraryapi
 * \ingroup module_selection
 */
#ifndef GMX_SELECTION_SELECTIONFILEOPTION_H
#define GMX_SELECTION_SELECTIONFILEOPTION_H

#include "gromacs/options/abstractoption.h"

namespace gmx
{

class SelectionFileOptionInfo;
class SelectionFileOptionStorage;
class SelectionOptionManager;

/*! \libinternal \brief
 * Specifies a special option that provides selections from a file.
 *
 * This option is used internally by the command-line framework to implement
 * file input for selections.  The option takes a file name, and reads it in
 * using SelectionOptionManager::parseRequestedFromFile().  This means that
 * selections from the file are assigned to selection options that have been
 * explicitly provided without values earlier on the command line.
 *
 * Public methods in this class do not throw.
 *
 * \inlibraryapi
 * \ingroup module_selection
 */
class SelectionFileOption : public AbstractOption
{
public:
    //! OptionInfo subclass corresponding to this option type.
    typedef SelectionFileOptionInfo InfoType;

    //! Initializes an option with the given name.
    explicit SelectionFileOption(const char* name);

private:
    AbstractOptionStorage* createStorage(const OptionManagerContainer& managers) const override;
};

/*! \libinternal \brief
 * Wrapper class for accessing and modifying selection file option information.
 *
 * \inlibraryapi
 * \ingroup module_selection
 */
class SelectionFileOptionInfo : public OptionInfo
{
public:
    /*! \brief
     * Creates option info object for given storage object.
     *
     * Does not throw.
     */
    explicit SelectionFileOptionInfo(SelectionFileOptionStorage* option);
};

} // namespace gmx

#endif
