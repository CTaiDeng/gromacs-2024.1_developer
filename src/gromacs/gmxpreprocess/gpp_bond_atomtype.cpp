/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 1991- The GROMACS Authors
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

#include "gmxpre.h"

#include "gpp_bond_atomtype.h"

#include <algorithm>
#include <optional>
#include <vector>

#include "gromacs/topology/symtab.h"
#include "gromacs/utility/smalloc.h"

class PreprocessingBondAtomType::Impl
{
public:
    //! The atom type names.
    std::vector<std::string> typeNames;
};

std::optional<int> PreprocessingBondAtomType::bondAtomTypeFromName(const std::string& str) const
{
    /* Atom types are always case sensitive */
    auto found = std::find_if(impl_->typeNames.begin(),
                              impl_->typeNames.end(),
                              [&str](const auto& type) { return str == std::string(type); });
    if (found == impl_->typeNames.end())
    {
        return std::nullopt;
    }
    else
    {
        return std::make_optional(std::distance(impl_->typeNames.begin(), found));
    }
}

std::optional<std::string> PreprocessingBondAtomType::atomNameFromBondAtomType(int nt) const
{
    return isSet(nt) ? impl_->typeNames[nt] : std::optional<std::string>{};
}

PreprocessingBondAtomType::PreprocessingBondAtomType() : impl_(new Impl) {}

PreprocessingBondAtomType::~PreprocessingBondAtomType() {}

int PreprocessingBondAtomType::addBondAtomType(const std::string& name)
{
    auto position = bondAtomTypeFromName(name);
    if (!position.has_value())
    {
        impl_->typeNames.emplace_back(name);
        if (auto bondAtomType = bondAtomTypeFromName(name); bondAtomType.has_value())
        {
            return *bondAtomType;
        }
        else
        {
            GMX_RELEASE_ASSERT(false, "Unhandled error in adding bond atom type");
            return 0;
        }
    }
    else
    {
        return *position;
    }
}

size_t PreprocessingBondAtomType::size() const
{
    return impl_->typeNames.size();
}

bool PreprocessingBondAtomType::isSet(int nt) const
{
    return ((nt >= 0) && (nt < gmx::ssize(*this)));
}
