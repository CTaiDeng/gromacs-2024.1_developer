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

#include "gmxapi/md.h"

#include <memory>

#include "gromacs/mdtypes/state.h"
#include "gromacs/utility/keyvaluetree.h"

#include "gmxapi/gmxapi.h"
#include "gmxapi/md/mdmodule.h"

#include "md_impl.h"

namespace gmxapi
{

const char* MDHolder::api_name = MDHolder_Name;

//! \cond internal
class MDWorkSpec::Impl
{
public:
    /*!
     * \brief container of objects glued together by the client.
     *
     * Note that the client can't be trusted to keep alive or destroy these,
     * and we do not yet have sufficient abstraction layers to allow the
     * client to interact asynchronously with code supporting MD simulation
     * through truly separate handles to the underlying objects, so ownership
     * is amongst the collaborators of the gmxapi::MDModule.
     * \todo Pass factory function objects instead of shared handles.
     * \todo Consolidate MDWorkSpec and gmxapi::Workflow under new Context umbrella.
     */
    std::vector<std::shared_ptr<gmxapi::MDModule>> modules;
};
//! \endcond

MDWorkSpec::MDWorkSpec() : impl_{ std::make_unique<Impl>() }
{
    GMX_ASSERT(impl_, "Expected non-null implementation object.");
}

void MDWorkSpec::addModule(std::shared_ptr<gmxapi::MDModule> module)
{
    GMX_ASSERT(impl_, "Expected non-null implementation object.");
    impl_->modules.emplace_back(std::move(module));
}

std::vector<std::shared_ptr<gmxapi::MDModule>>& MDWorkSpec::getModules()
{
    GMX_ASSERT(impl_, "Expected non-null implementation object.");
    return impl_->modules;
}

MDWorkSpec::~MDWorkSpec() = default;

std::shared_ptr<::gmxapi::MDWorkSpec> MDHolder::getSpec()
{
    GMX_ASSERT(impl_, "Expected non-null implementation object.");
    GMX_ASSERT(impl_->spec_, "Expected non-null work specification.");
    return impl_->spec_;
}

std::shared_ptr<const ::gmxapi::MDWorkSpec> MDHolder::getSpec() const
{
    GMX_ASSERT(impl_, "Expected non-null implementation object.");
    GMX_ASSERT(impl_->spec_, "Expected non-null work specification.");
    return impl_->spec_;
}

MDHolder::MDHolder() : MDHolder{ std::make_shared<MDWorkSpec>() }
{
    GMX_ASSERT(impl_, "Expected non-null implementation object.");
    GMX_ASSERT(impl_->spec_, "Expected non-null work specification.");
}

MDHolder::MDHolder(std::shared_ptr<MDWorkSpec> spec) :
    impl_(std::make_shared<MDHolder::Impl>(std::move(spec)))
{
    GMX_ASSERT(impl_, "Expected non-null implementation object.");
    GMX_ASSERT(impl_->spec_, "Expected non-null work specification.");
}

MDHolder::Impl::Impl(std::shared_ptr<MDWorkSpec>&& spec) : spec_{ spec }
{
    GMX_ASSERT(spec_, "Expected non-null work specification.");
}

MDHolder::MDHolder(std::string name) : MDHolder{}
{
    name_ = std::move(name);
    GMX_ASSERT(impl_, "Expected non-null implementation object.");
    GMX_ASSERT(impl_->spec_, "Expected non-null work specification.");
}

std::string MDHolder::name() const
{
    return name_;
}


} // end namespace gmxapi
