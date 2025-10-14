/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2016- The GROMACS Authors
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
 * Declares gmx::IOptionSectionStorage.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_options
 */
#ifndef GMX_OPTIONS_ISECTIONSTORAGE_H
#define GMX_OPTIONS_ISECTIONSTORAGE_H

namespace gmx
{

/*! \internal
 * \brief
 * Provides behavior specific to a certain option section type.
 *
 * \ingroup module_options
 */
class IOptionSectionStorage
{
public:
    virtual ~IOptionSectionStorage();

    /*! \brief
     * Called once before the first call to startSection().
     *
     * This is called once all options have been added to the section.
     * The current implementation does not call this if startSection() is
     * never called.
     */
    virtual void initStorage() = 0;
    /*! \brief
     * Called when option assignment enters this section.
     */
    virtual void startSection() = 0;
    /*! \brief
     * Called when option assignment leaves this section.
     */
    virtual void finishSection() = 0;
};

} // namespace gmx

#endif
