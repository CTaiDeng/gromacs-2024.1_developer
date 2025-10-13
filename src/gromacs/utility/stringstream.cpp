/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2015- The GROMACS Authors
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
 * Implements classes from stringstream.h.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_utility
 */
#include "gmxpre.h"

#include "gromacs/utility/stringstream.h"

#include <string>

#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/stringutil.h"

namespace gmx
{

void StringOutputStream::write(const char* str)
{
    str_.append(str);
}

void StringOutputStream::close() {}

StringInputStream::StringInputStream(const std::string& input) : input_(input), pos_(0) {}

StringInputStream::StringInputStream(const std::vector<std::string>& input) :
    input_(joinStrings(input.begin(), input.end(), "\n")), pos_(0)
{
    input_.append("\n");
}

StringInputStream::StringInputStream(ArrayRef<const char* const> const& input) :
    input_(joinStrings(input.begin(), input.end(), "\n")), pos_(0)
{
    input_.append("\n");
}

bool StringInputStream::readLine(std::string* line)
{
    if (pos_ == input_.size())
    {
        line->clear();
        return false;
    }
    else
    {
        size_t newpos = input_.find('\n', pos_);
        if (newpos == std::string::npos)
        {
            newpos = input_.size();
        }
        else
        {
            // To include the newline as well!
            newpos += 1;
        }
        line->assign(input_.substr(pos_, newpos - pos_));
        pos_ = newpos;
        return true;
    }
}

} // namespace gmx
