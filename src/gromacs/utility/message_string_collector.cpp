/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2011- The GROMACS Authors
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
 * Implements gmx::MessageStringCollector.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_utility
 */
#include "gmxpre.h"

#include "gromacs/utility/message_string_collector.h"

#include <vector>

#include "gromacs/utility/gmxassert.h"

namespace gmx
{

class MessageStringCollector::Impl
{
public:
    Impl() : prevContext_(0) {}

    std::vector<std::string> contexts_;
    std::string              text_;
    size_t                   prevContext_;
};

MessageStringCollector::MessageStringCollector() : impl_(new Impl) {}

MessageStringCollector::~MessageStringCollector()                                 = default;
MessageStringCollector::MessageStringCollector(MessageStringCollector&&) noexcept = default;
MessageStringCollector& MessageStringCollector::operator=(MessageStringCollector&&) noexcept = default;

void MessageStringCollector::startContext(const char* name)
{
    impl_->contexts_.emplace_back(name);
}

void MessageStringCollector::append(const std::string& message)
{
    int indent = static_cast<int>(impl_->prevContext_ * 2);
    if (!impl_->contexts_.empty())
    {
        std::vector<std::string>::const_iterator ci;
        for (ci = impl_->contexts_.begin() + impl_->prevContext_; ci != impl_->contexts_.end(); ++ci)
        {
            impl_->text_.append(indent, ' ');
            impl_->text_.append(*ci);
            impl_->text_.append("\n");
            indent += 2;
        }
    }
    impl_->prevContext_ = impl_->contexts_.size();

    // TODO: Put this into a more generic helper, could be useful elsewhere
    size_t pos = 0;
    while (pos < message.size())
    {
        size_t nextpos = message.find_first_of('\n', pos);
        impl_->text_.append(indent, ' ');
        impl_->text_.append(message.substr(pos, nextpos - pos));
        impl_->text_.append("\n");
        if (nextpos == std::string::npos)
        {
            break;
        }
        pos = nextpos + 1;
    }
}

void MessageStringCollector::appendIf(bool condition, const char* message)
{
    if (condition)
    {
        append(std::string(message));
    }
}

void MessageStringCollector::appendIf(bool condition, const std::string& message)
{
    if (condition)
    {
        append(message);
    }
}

void MessageStringCollector::finishContext()
{
    GMX_RELEASE_ASSERT(!impl_->contexts_.empty(), "finishContext() called without context");
    impl_->contexts_.pop_back();
    if (impl_->prevContext_ > impl_->contexts_.size())
    {
        impl_->prevContext_ = impl_->contexts_.size();
    }
}

void MessageStringCollector::clear()
{
    impl_->contexts_.clear();
    impl_->text_.clear();
    impl_->prevContext_ = 0;
}

bool MessageStringCollector::isEmpty() const
{
    return impl_->text_.empty();
}

std::string MessageStringCollector::toString() const
{
    return impl_->text_;
}

} // namespace gmx
