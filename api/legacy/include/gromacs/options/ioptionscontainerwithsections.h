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
 * Declares gmx::IOptionsContainerWithSections.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \inpublicapi
 * \ingroup module_options
 */
#ifndef GMX_OPTIONS_IOPTIONSCONTAINERWITHSECTIONS_H
#define GMX_OPTIONS_IOPTIONSCONTAINERWITHSECTIONS_H

#include "gromacs/options/ioptionscontainer.h"

namespace gmx
{

class AbstractOptionSection;
class AbstractOptionSectionHandle;

namespace internal
{
class OptionSectionImpl;
}

/*! \brief
 * Interface for adding input options with sections.
 *
 * This interface extends IOptionsContainer with an additional addSection()
 * method that supports creating a hierarchy of sections for the options.
 *
 * Header optionsection.h provides OptionSection.
 *
 * \inpublicapi
 * \ingroup module_options
 */
class IOptionsContainerWithSections : public IOptionsContainer
{
public:
    /*! \brief
     * Adds a section to this collection.
     *
     * \tparam    SectionType Type of the section description object.
     * \param[in] section     Section description.
     * \returns   AbstractOptionSectionHandle object for the created option.
     * \throws    APIError if invalid option settings are provided.
     *
     * Options can be added to the section through the returned handle.
     *
     * \internal
     * \p SectionType::HandleType must specify a type that derives from
     * AbstractinOptionSectionHandle and has a suitable constructor.
     */
    template<class SectionType>
    typename SectionType::HandleType addSection(const SectionType& section)
    {
        internal::OptionSectionImpl* storage =
                addSectionImpl(static_cast<const AbstractOptionSection&>(section));
        return typename SectionType::HandleType(storage);
    }

protected:
    // Disallow deletion through the interface.
    // (no need for the virtual, but some compilers warn otherwise)
    ~IOptionsContainerWithSections() override;

    /*! \brief
     * Adds a section to this container.
     *
     * \param[in] section     Section description.
     * \returns   Pointer to the internal section representation object.
     */
    virtual internal::OptionSectionImpl* addSectionImpl(const AbstractOptionSection& section) = 0;

    GMX_DEFAULT_CONSTRUCTORS(IOptionsContainerWithSections);
};

} // namespace gmx

#endif
