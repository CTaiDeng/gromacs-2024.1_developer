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

/*! \internal \file
 * \brief
 * Implements classes in LocalAtomSetmanager.h.
 *
 * \author Christian Blau <cblau@gwdg.de>
 * \ingroup module_domdec
 */
#include "gmxpre.h"

#include "localatomsetmanager.h"

#include <algorithm>
#include <memory>

#include "gromacs/domdec/localatomset.h"
#include "gromacs/utility/exceptions.h"

#include "localatomsetdata.h"

namespace gmx
{

/********************************************************************
 * LocalAtomSetManager::Impl */

/*! \internal \brief
 * Private implementation class for LocalAtomSetManager.
 */
class LocalAtomSetManager::Impl
{
public:
    std::vector<std::unique_ptr<internal::LocalAtomSetData>> atomSetData_; /**< handles to the managed atom sets */
};

/********************************************************************
 * LocalAtomSetManager */

LocalAtomSetManager::LocalAtomSetManager() : impl_(new Impl()) {}

LocalAtomSetManager::~LocalAtomSetManager() {}

template<>
LocalAtomSet LocalAtomSetManager::add<void, void>(ArrayRef<const int> globalAtomIndex)
{
    impl_->atomSetData_.push_back(std::make_unique<internal::LocalAtomSetData>(globalAtomIndex));
    return LocalAtomSet(*impl_->atomSetData_.back());
}

LocalAtomSet LocalAtomSetManager::add(ArrayRef<const Index> globalAtomIndex)
{
    impl_->atomSetData_.push_back(std::make_unique<internal::LocalAtomSetData>(globalAtomIndex));
    return LocalAtomSet(*impl_->atomSetData_.back());
}

void LocalAtomSetManager::setIndicesInDomainDecomposition(const gmx_ga2la_t& ga2la)
{
    for (const auto& atomSet : impl_->atomSetData_)
    {
        atomSet->setLocalAndCollectiveIndices(ga2la);
    }
}

} // namespace gmx
