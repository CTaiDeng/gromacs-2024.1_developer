/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2016- The GROMACS Authors
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
 * Declares gmx::OptionSection and gmx::OptionSectionInfo.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \inpublicapi
 * \ingroup module_options
 */
#ifndef GMX_OPTIONS_OPTIONSECTION_H
#define GMX_OPTIONS_OPTIONSECTION_H

#include <memory>

#include "abstractsection.h"

namespace gmx
{

class OptionSectionHandle;

/*! \brief
 * Declares a simple option section.
 *
 * This class declares a simple section that only provides structure for
 * grouping the options, but does not otherwise influence the behavior of the
 * contained options.
 *
 * \inpublicapi
 * \ingroup module_options
 */
class OptionSection : public AbstractOptionSection
{
public:
    //! AbstractOptionSectionHandle corresponding to this option type.
    typedef OptionSectionHandle HandleType;

    //! Creates a section with the given name.
    explicit OptionSection(const char* name) : AbstractOptionSection(name) {}

private:
    std::unique_ptr<IOptionSectionStorage> createStorage() const override;
};

/*! \brief
 * Allows adding options to an OptionSection.
 *
 * An instance of this class is returned from
 * IOptionsContainerWithSections::addSection(), and supports adding options and
 * subsections to a section created with OptionSection.
 *
 * \inpublicapi
 * \ingroup module_options
 */
class OptionSectionHandle : public AbstractOptionSectionHandle
{
public:
    //! Wraps a given section storage object.
    explicit OptionSectionHandle(internal::OptionSectionImpl* section) :
        AbstractOptionSectionHandle(section)
    {
    }
};

class OptionSectionInfo : public AbstractOptionSectionInfo
{
public:
    //! Wraps a given section storage object.
    explicit OptionSectionInfo(internal::OptionSectionImpl* section) :
        AbstractOptionSectionInfo(section)
    {
    }
};

} // namespace gmx

#endif
