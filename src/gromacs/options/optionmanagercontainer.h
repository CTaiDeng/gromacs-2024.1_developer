/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2014- The GROMACS Authors
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
 * Declares gmx::OptionManagerContainer.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \inlibraryapi
 * \ingroup module_options
 */
#ifndef GMX_OPTIONS_OPTIONMANAGERCONTAINER_H
#define GMX_OPTIONS_OPTIONMANAGERCONTAINER_H

#include <memory>
#include <vector>

#include "gromacs/utility/classhelpers.h"
#include "gromacs/utility/gmxassert.h"

namespace gmx
{

class IOptionManager;

/*! \libinternal
 * \brief
 * Container to keep managers added with Options::addManager() and pass them
 * to options.
 *
 * Consistency of the managers (e.g., that there is at most one manager of a
 * certain type) is only checked when the managers are accessed.
 *
 * \inlibraryapi
 * \ingroup module_options
 */
class OptionManagerContainer
{
public:
    OptionManagerContainer() {}

    //! Returns `true` if there are no managers.
    bool empty() const { return list_.empty(); }

    //! Adds a manager to the container.
    void add(IOptionManager* manager) { list_.push_back(manager); }
    /*! \brief
     * Retrieves a manager of a certain type.
     *
     * \tparam  ManagerType  Type of manager to retrieve
     *     (should derive from IOptionManager).
     * \returns The manager, or `NULL` if there is none.
     *
     * This method is used in AbstractOption::createStorage() to retrieve
     * a manager of a certain type for options that use a manager.
     *
     * The return value is `NULL` if there is no manager of the given type.
     * The caller needs to handle this (either by asserting, or by handling
     * the manager as optional).
     */
    template<class ManagerType>
    ManagerType* get() const
    {
        ManagerType* result = nullptr;
        for (const auto& i : list_)
        {
            ManagerType* curr = dynamic_cast<ManagerType*>(i);
            if (curr != nullptr)
            {
                GMX_RELEASE_ASSERT(result == nullptr,
                                   "More than one applicable option manager is set");
                result = curr;
            }
        }
        return result;
    }

private:
    //! Shorthand for the internal container type.
    typedef std::vector<IOptionManager*> ListType;

    ListType list_;

    GMX_DISALLOW_COPY_AND_ASSIGN(OptionManagerContainer);
};

} // namespace gmx

#endif
