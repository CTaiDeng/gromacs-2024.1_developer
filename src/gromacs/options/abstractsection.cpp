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

/*! \internal \file
 * \brief
 * Implements classes from abstractsection.h.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_options
 */
#include "gmxpre.h"

#include "abstractsection.h"

#include "options_impl.h"

namespace gmx
{

/********************************************************************
 * AbstractOptionSectionHandle
 */

// static
IOptionSectionStorage* AbstractOptionSectionHandle::getStorage(internal::OptionSectionImpl* section)
{
    return section->storage_.get();
}

IOptionsContainer& AbstractOptionSectionHandle::addGroup()
{
    return section_->addGroup();
}

internal::OptionSectionImpl* AbstractOptionSectionHandle::addSectionImpl(const AbstractOptionSection& section)
{
    return section_->addSectionImpl(section);
}

OptionInfo* AbstractOptionSectionHandle::addOptionImpl(const AbstractOption& settings)
{
    return section_->addOptionImpl(settings);
}

/********************************************************************
 * AbstractOptionSectionInfo
 */

const std::string& AbstractOptionSectionInfo::name() const
{
    return section_.name_;
}

} // namespace gmx
