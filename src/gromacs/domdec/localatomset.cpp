/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2018- The GROMACS Authors
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

/*! \file
 * \internal \brief
 * Implements classes in LocalAtomSet.h
 *
 * \author Christian Blau <cblau@gwdg.de>
 * \ingroup module_domdec
 */
#include "gmxpre.h"

#include "localatomset.h"

#include "localatomsetdata.h"

namespace gmx
{

LocalAtomSet::LocalAtomSet(const internal::LocalAtomSetData& data) : data_(&data) {}

ArrayRef<const int> LocalAtomSet::globalIndex() const
{
    return data_->globalIndex_;
}

ArrayRef<const int> LocalAtomSet::localIndex() const
{
    return data_->localIndex_;
}

ArrayRef<const int> LocalAtomSet::collectiveIndex() const
{
    return data_->collectiveIndex_;
}

std::size_t LocalAtomSet::numAtomsGlobal() const
{
    return data_->globalIndex_.size();
}

std::size_t LocalAtomSet::numAtomsLocal() const
{
    return data_->localIndex_.size();
}

} // namespace gmx
