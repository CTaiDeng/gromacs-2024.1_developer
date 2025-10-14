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
 * Implements classes in abstractoption.h and abstractoptionstorage.h.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_options
 */
#include "gmxpre.h"

#include "gromacs/options/abstractoption.h"

#include "gromacs/options/abstractoptionstorage.h"
#include "gromacs/options/optionflags.h"
#include "gromacs/utility/any.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/gmxassert.h"

#include "basicoptionstorage.h"

namespace gmx
{

/********************************************************************
 * AbstractOptionStorage
 */

AbstractOptionStorage::AbstractOptionStorage(const AbstractOption& settings, OptionFlags staticFlags) :
    flags_(settings.flags_ | staticFlags),
    storeIsSet_(settings.storeIsSet_),
    minValueCount_(settings.minValueCount_),
    maxValueCount_(settings.maxValueCount_),
    bInSet_(false),
    bSetValuesHadErrors_(false)
{
    // Check that user has not provided incorrect values for vectors.
    if (hasFlag(efOption_Vector) && (minValueCount_ > 1 || maxValueCount_ < 1))
    {
        GMX_THROW(APIError("Inconsistent value counts for vector values"));
    }

    if (settings.name_ != nullptr)
    {
        name_ = settings.name_;
    }
    if (settings.descr_ != nullptr)
    {
        descr_ = settings.descr_;
    }
    if (storeIsSet_ != nullptr)
    {
        *storeIsSet_ = false;
    }
    setFlag(efOption_ClearOnNextSet);
}

AbstractOptionStorage::~AbstractOptionStorage() {}

bool AbstractOptionStorage::isBoolean() const
{
    return dynamic_cast<const BooleanOptionStorage*>(this) != nullptr;
}

void AbstractOptionStorage::startSource()
{
    setFlag(efOption_ClearOnNextSet);
}

void AbstractOptionStorage::startSet()
{
    GMX_RELEASE_ASSERT(!bInSet_, "finishSet() not called");
    // The last condition takes care of the situation where multiple
    // sources are used, and a later source should be able to reassign
    // the value even though the option is already set.
    if (isSet() && !hasFlag(efOption_MultipleTimes) && !hasFlag(efOption_ClearOnNextSet))
    {
        GMX_THROW(InvalidInputError("Option specified multiple times"));
    }
    clearSet();
    bInSet_              = true;
    bSetValuesHadErrors_ = false;
}

void AbstractOptionStorage::appendValue(const Any& value)
{
    GMX_RELEASE_ASSERT(bInSet_, "startSet() not called");
    try
    {
        convertValue(value);
    }
    catch (const std::exception&)
    {
        bSetValuesHadErrors_ = true;
        throw;
    }
}

void AbstractOptionStorage::markAsSet()
{
    setFlag(efOption_Set);
    if (storeIsSet_ != nullptr)
    {
        *storeIsSet_ = true;
    }
}

void AbstractOptionStorage::finishSet()
{
    GMX_RELEASE_ASSERT(bInSet_, "startSet() not called");
    bInSet_ = false;
    // We mark the option as set even when there are errors to avoid additional
    // errors from required options not set.
    // TODO: There could be a separate flag for this purpose.
    markAsSet();
    if (!bSetValuesHadErrors_)
    {
        // TODO: Correct handling of the efOption_ClearOnNextSet requires
        // processSet() and/or convertValue() to check it internally.
        // OptionStorageTemplate takes care of it, but it's error-prone if
        // a custom option is implemented that doesn't use it.
        processSet();
    }
    bSetValuesHadErrors_ = false;
    clearFlag(efOption_ClearOnNextSet);
    clearSet();
}

void AbstractOptionStorage::finish()
{
    GMX_RELEASE_ASSERT(!bInSet_, "finishSet() not called");
    processAll();
    if (isRequired() && !(isSet() || hasFlag(efOption_ExplicitDefaultValue)))
    {
        GMX_THROW(InvalidInputError("Option is required, but not set"));
    }
}

void AbstractOptionStorage::setMinValueCount(int count)
{
    GMX_RELEASE_ASSERT(!hasFlag(efOption_MultipleTimes),
                       "setMinValueCount() not supported with efOption_MultipleTimes");
    GMX_RELEASE_ASSERT(count >= 0, "Invalid value count");
    minValueCount_ = count;
    if (isSet() && !hasFlag(efOption_DontCheckMinimumCount) && valueCount() < minValueCount_)
    {
        GMX_THROW(InvalidInputError("Too few values"));
    }
}

void AbstractOptionStorage::setMaxValueCount(int count)
{
    GMX_RELEASE_ASSERT(!hasFlag(efOption_MultipleTimes),
                       "setMaxValueCount() not supported with efOption_MultipleTimes");
    GMX_RELEASE_ASSERT(count >= -1, "Invalid value count");
    maxValueCount_ = count;
    if (isSet() && maxValueCount_ >= 0 && valueCount() > maxValueCount_)
    {
        GMX_THROW(InvalidInputError("Too many values"));
    }
}

/********************************************************************
 * OptionInfo
 */

/*! \cond libapi */
OptionInfo::OptionInfo(AbstractOptionStorage* option) : option_(*option) {}
//! \endcond

OptionInfo::~OptionInfo() {}

bool OptionInfo::isSet() const
{
    return option().isSet();
}

bool OptionInfo::isHidden() const
{
    return option().isHidden();
}

bool OptionInfo::isRequired() const
{
    return option().isRequired();
}

int OptionInfo::minValueCount() const
{
    if (option().defaultValueIfSetExists())
    {
        return 0;
    }
    return option().minValueCount();
}

int OptionInfo::maxValueCount() const
{
    return option().maxValueCount();
}

const std::string& OptionInfo::name() const
{
    return option().name();
}

std::string OptionInfo::type() const
{
    return option().typeString();
}

std::string OptionInfo::formatDescription() const
{
    std::string description(option().description());
    std::string extraDescription(option().formatExtraDescription());
    if (!extraDescription.empty())
    {
        description.append(extraDescription);
    }
    return description;
}

std::vector<Any> OptionInfo::defaultValues() const
{
    return option().defaultValues();
}

std::vector<std::string> OptionInfo::defaultValuesAsStrings() const
{
    return option().defaultValuesAsStrings();
}

std::vector<Any> OptionInfo::normalizeValues(const std::vector<Any>& values) const
{
    return option().normalizeValues(values);
}

} // namespace gmx
