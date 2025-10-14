/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2015- The GROMACS Authors
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
 * Declares gmx::OptionsBehaviorCollection.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \inlibraryapi
 * \ingroup module_options
 */
#ifndef GMX_OPTIONS_BEHAVIORCOLLECTION_H
#define GMX_OPTIONS_BEHAVIORCOLLECTION_H

#include <memory>
#include <vector>

#include "gromacs/utility/classhelpers.h"

namespace gmx
{

class IOptionsBehavior;
class Options;

//! Smart pointer for behaviors stored in OptionsBehaviorCollection.
typedef std::shared_ptr<IOptionsBehavior> OptionsBehaviorPointer;

/*! \libinternal \brief
 * Container for IOptionsBehavior objects.
 *
 * This class provides a container to keep IOptionsBehavior objects, and to
 * call the IOptionsBehavior methods for the contained objects.
 *
 * IOptionsBehavior methods are called for the contained objects in the same
 * order as in which the behaviors were inserted.
 *
 * \inlibraryapi
 * \ingroup module_options
 */
class OptionsBehaviorCollection
{
public:
    /*! \brief
     * Constructs a container for storing behaviors associated with given
     * Options.
     *
     * Caller needs to ensure that provided Options remains in existence
     * while the container exists.
     */
    explicit OptionsBehaviorCollection(Options* options);
    ~OptionsBehaviorCollection();

    //! Adds a behavior to the collection.
    void addBehavior(const OptionsBehaviorPointer& behavior);
    //! Calls IOptionsBehavior::optionsFinishing() on all behaviors.
    void optionsFinishing();
    //! Calls IOptionsBehavior::optionsFinished() on all behaviors.
    void optionsFinished();

private:
    Options*                            options_;
    std::vector<OptionsBehaviorPointer> behaviors_;

    GMX_DISALLOW_COPY_AND_ASSIGN(OptionsBehaviorCollection);
};

} // namespace gmx

#endif
