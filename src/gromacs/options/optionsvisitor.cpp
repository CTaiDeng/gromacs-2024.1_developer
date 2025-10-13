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

/*! \internal \file
 * \brief
 * Implements classes in optionsvisitor.h.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_options
 */
#include "gmxpre.h"

#include "optionsvisitor.h"

#include "gromacs/options/abstractoptionstorage.h"
#include "gromacs/options/options.h"
#include "gromacs/options/optionsection.h"

#include "options_impl.h"

namespace gmx
{

namespace
{

//! Helper function to call visitOptions() and handle correct indirection.
void visitOption(OptionsVisitor* visitor, OptionInfo& optionInfo) //NOLINT(google-runtime-references)
{
    visitor->visitOption(optionInfo);
}
//! Helper function to call visitOptions() and handle correct indirection.
void visitOption(OptionsModifyingVisitor* visitor, OptionInfo& optionInfo) //NOLINT(google-runtime-references)
{
    visitor->visitOption(&optionInfo);
}

//! Helper function to recursively visit all options in a group.
template<class VisitorType>
void acceptOptionsGroup(const internal::OptionSectionImpl::Group& group, VisitorType* visitor)
{
    for (const auto& option : group.options_)
    {
        visitOption(visitor, option->optionInfo());
    }
    for (const auto& subgroup : group.subgroups_)
    {
        acceptOptionsGroup(subgroup, visitor);
    }
}

} // namespace

/********************************************************************
 * OptionsIterator
 */

OptionsIterator::OptionsIterator(const Options& options) : section_(options.rootSection().section())
{
}

OptionsIterator::OptionsIterator(const OptionSectionInfo& section) : section_(section.section()) {}

void OptionsIterator::acceptSections(OptionsVisitor* visitor) const
{
    for (const auto& section : section_.subsections_)
    {
        visitor->visitSection(section->info());
    }
}

void OptionsIterator::acceptOptions(OptionsVisitor* visitor) const
{
    acceptOptionsGroup(section_.rootGroup_, visitor);
}

/********************************************************************
 * OptionsModifyingIterator
 */

OptionsModifyingIterator::OptionsModifyingIterator(Options* options) :
    section_(options->rootSection().section())
{
}

OptionsModifyingIterator::OptionsModifyingIterator(OptionSectionInfo* section) :
    section_(section->section())
{
}

void OptionsModifyingIterator::acceptSections(OptionsModifyingVisitor* visitor) const
{
    for (auto& section : section_.subsections_)
    {
        visitor->visitSection(&section->info());
    }
}

void OptionsModifyingIterator::acceptOptions(OptionsModifyingVisitor* visitor) const
{
    acceptOptionsGroup(section_.rootGroup_, visitor);
}

} // namespace gmx
