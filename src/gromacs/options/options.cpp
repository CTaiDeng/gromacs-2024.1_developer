/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2010- The GROMACS Authors
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
 * Implements gmx::Options.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_options
 */
#include "gmxpre.h"

#include "gromacs/options/options.h"

#include <memory>
#include <utility>

#include "gromacs/options/abstractoption.h"
#include "gromacs/options/abstractoptionstorage.h"
#include "gromacs/options/optionsection.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/stringutil.h"

#include "options_impl.h"

namespace gmx
{

/********************************************************************
 * IOptionManager
 */

IOptionManager::~IOptionManager() {}

/********************************************************************
 * IOptionsContainer
 */

IOptionsContainer::~IOptionsContainer() {}

/********************************************************************
 * IOptionsContainerWithSections
 */

IOptionsContainerWithSections::~IOptionsContainerWithSections() {}

/********************************************************************
 * IOptionSectionStorage
 */

IOptionSectionStorage::~IOptionSectionStorage() {}

/********************************************************************
 * OptionsImpl
 */

namespace internal
{

OptionsImpl::OptionsImpl() : rootSection_(managers_, nullptr, "") {}

/********************************************************************
 * OptionSectionImpl
 */

OptionSectionImpl* OptionSectionImpl::addSectionImpl(const AbstractOptionSection& section)
{
    const char* name = section.name_;
    // Make sure that there are no duplicate sections.
    GMX_RELEASE_ASSERT(findSection(name) == nullptr, "Duplicate subsection name");
    std::unique_ptr<IOptionSectionStorage> storage(section.createStorage());
    subsections_.push_back(std::make_unique<OptionSectionImpl>(managers_, std::move(storage), name));
    return subsections_.back().get();
}

IOptionsContainer& OptionSectionImpl::addGroup()
{
    return rootGroup_.addGroup();
}

OptionInfo* OptionSectionImpl::addOptionImpl(const AbstractOption& settings)
{
    return rootGroup_.addOptionImpl(settings);
}

OptionSectionImpl* OptionSectionImpl::findSection(const char* name) const
{
    for (const auto& section : subsections_)
    {
        if (section->name_ == name)
        {
            return section.get();
        }
    }
    return nullptr;
}

AbstractOptionStorage* OptionSectionImpl::findOption(const char* name) const
{
    OptionMap::const_iterator i = optionMap_.find(name);
    if (i == optionMap_.end())
    {
        return nullptr;
    }
    return i->second.get();
}

void OptionSectionImpl::start()
{
    for (const auto& entry : optionMap_)
    {
        entry.second->startSource();
    }
    if (storage_ != nullptr)
    {
        if (!storageInitialized_)
        {
            storage_->initStorage();
            storageInitialized_ = true;
        }
        storage_->startSection();
    }
}

void OptionSectionImpl::finish()
{
    // TODO: Consider how to customize these error messages based on context.
    ExceptionInitializer errors("Invalid input values");
    for (const auto& entry : optionMap_)
    {
        AbstractOptionStorage& option = *entry.second;
        try
        {
            option.finish();
        }
        catch (UserInputError& ex)
        {
            ex.prependContext("In option " + option.name());
            errors.addCurrentExceptionAsNested();
        }
    }
    if (errors.hasNestedExceptions())
    {
        // TODO: This exception type may not always be appropriate.
        GMX_THROW(InvalidInputError(errors));
    }
    if (storage_ != nullptr)
    {
        storage_->finishSection();
    }
}

/********************************************************************
 * OptionSectionImpl::Group
 */

IOptionsContainer& OptionSectionImpl::Group::addGroup()
{
    subgroups_.emplace_back(parent_);
    return subgroups_.back();
}

OptionInfo* OptionSectionImpl::Group::addOptionImpl(const AbstractOption& settings)
{
    OptionSectionImpl::AbstractOptionStoragePointer option(settings.createStorage(parent_->managers_));
    options_.reserve(options_.size() + 1);
    auto insertionResult = parent_->optionMap_.insert(std::make_pair(option->name(), std::move(option)));
    if (!insertionResult.second)
    {
        const std::string& name = insertionResult.first->second->name();
        GMX_THROW(APIError("Duplicate option: " + name));
    }
    AbstractOptionStorage& insertedOption = *insertionResult.first->second;
    options_.push_back(&insertedOption);
    return &insertedOption.optionInfo();
}

} // namespace internal

using internal::OptionsImpl;

/********************************************************************
 * Options
 */

Options::Options() : impl_(new OptionsImpl) {}

Options::~Options() {}


void Options::addManager(IOptionManager* manager)
{
    // This ensures that all options see the same set of managers.
    GMX_RELEASE_ASSERT(impl_->rootSection_.optionMap_.empty(),
                       "Can only add a manager before options");
    // This check could be relaxed if we instead checked that the subsections
    // do not have options.
    GMX_RELEASE_ASSERT(impl_->rootSection_.subsections_.empty(),
                       "Can only add a manager before subsections");
    impl_->managers_.add(manager);
}

internal::OptionSectionImpl* Options::addSectionImpl(const AbstractOptionSection& section)
{
    return impl_->rootSection_.addSectionImpl(section);
}

IOptionsContainer& Options::addGroup()
{
    return impl_->rootSection_.addGroup();
}

OptionInfo* Options::addOptionImpl(const AbstractOption& settings)
{
    return impl_->rootSection_.addOptionImpl(settings);
}

OptionSectionInfo& Options::rootSection()
{
    return impl_->rootSection_.info();
}

const OptionSectionInfo& Options::rootSection() const
{
    return impl_->rootSection_.info();
}

void Options::finish()
{
    impl_->rootSection_.finish();
}

} // namespace gmx
