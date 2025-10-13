/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2020- The GROMACS Authors
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

#include "gmxpre.h"

#include "simulationinputhandle.h"

#include <utility>

#include "gromacs/mdrun/legacymdrunoptions.h"

namespace gmx
{

/*! \cond
 * SimulationInput needs much more discussion and refinement before it should be used directly.
 * Use the interface defined in the headers and refer to #3652 for updates.
 *
 * Design note: It is probably sufficient for future versions to compose SimulationInput
 * through a Builder rather than to subclass an Interface or base class. Outside of this
 * translation unit, we should avoid coupling to the class definition until/unless we
 * develop a much better understanding of simulation input portability.
 *
 */
class SimulationInput
{
public:
    SimulationInput(const char* tprFilename, const char* cpiFilename) :
        tprFilename_(tprFilename), cpiFilename_(cpiFilename)
    {
    }

    std::string tprFilename_;
    std::string cpiFilename_;
};
/*! \endcond */

class detail::SimulationInputHandleImpl final
{
public:
    explicit SimulationInputHandleImpl(std::shared_ptr<SimulationInput> input) :
        simulationInput_{ std::move(input) }
    {
    }

    SimulationInputHandleImpl(const SimulationInputHandleImpl&) = default;
    SimulationInputHandleImpl& operator=(const SimulationInputHandleImpl&) = default;
    SimulationInputHandleImpl(SimulationInputHandleImpl&&) noexcept        = default;
    SimulationInputHandleImpl& operator=(SimulationInputHandleImpl&&) noexcept = default;

    [[nodiscard]] SimulationInput* simulationInput() const { return simulationInput_.get(); }

    /*! \brief Boolean interpretation checks for presence of payload.
     *
     * \return true if handle has a SimulationInput, else false.
     */
    operator bool() const { return bool(simulationInput_); }

private:
    // Note that we should not need to worry about managing the SimulationInput
    // Deleter as long as we manage the SimulationInputHolderImpl Deleter and
    // SimulationInputHolderImpl is the only way to own a SimulationInput.
    // BUT these semantics may change, and we should try to be responsible about
    // consistent Allocator implementation.
    std::shared_ptr<SimulationInput> simulationInput_;
};

/*! \cond
 *  TODO: Improve doxygen source scoping.
 *  It seems surprising that doxygen should be looking at this source file, but we
 *  need to exclude the following definitions to avoid warnings.
 *  Possibly related to https://gitlab.com/gromacs/gromacs/-/issues/3402
 */
detail::SimulationInputHandleImplDeleter::SimulationInputHandleImplDeleter() = default;

detail::SimulationInputHandleImplDeleter::SimulationInputHandleImplDeleter(
        const SimulationInputHandleImplDeleter&) noexcept = default;

detail::SimulationInputHandleImplDeleter::SimulationInputHandleImplDeleter(
        SimulationInputHandleImplDeleter&&) noexcept = default;

detail::SimulationInputHandleImplDeleter& detail::SimulationInputHandleImplDeleter::operator=(
        const SimulationInputHandleImplDeleter&) noexcept = default;

detail::SimulationInputHandleImplDeleter& detail::SimulationInputHandleImplDeleter::operator=(
        SimulationInputHandleImplDeleter&&) noexcept = default;

void detail::SimulationInputHandleImplDeleter::operator()(SimulationInputHandleImpl* impl) const
{
    delete impl;
}

SimulationInputHandle::SimulationInputHandle() = default;

SimulationInputHandle::~SimulationInputHandle() = default;

SimulationInputHandle::SimulationInputHandle(std::unique_ptr<detail::SimulationInputHandleImpl> impl) :
    impl_(impl.release())
{
}

SimulationInputHandle::SimulationInputHandle(const SimulationInputHandle& source)
{
    // Dynamically allocate a new holder implementation object using the copy constructor.
    // Note that make_unique does not provide for custom deleters, and there is no default
    // overload to move construct/assign from basic unique_ptr to one with custom deleter,
    // but more elaborate creation functions for SimulationInputHolderImpl seem like
    // overkill at this point. Still, we use make_unique to confine constructor errors
    // to the following exception-safe line.
    auto impl = std::make_unique<detail::SimulationInputHandleImpl>(*source.impl_);
    // Move ownership of the heap object to the handle with a custom deleter.
    impl_.reset(impl.release());
}

SimulationInputHandle& SimulationInputHandle::operator=(const SimulationInputHandle& source)
{
    if (this != &source)
    {
        // Dynamically allocate a new holder implementation object using the copy constructor.
        // Note that make_unique does not provide for custom deleters, and there is no default
        // overload to move construct/assign from basic unique_ptr to one with custom deleter,
        // but more elaborate creation functions for SimulationInputHolderImpl seem like
        // overkill at this point. Still, we use make_unique to confine constructor errors
        // to the following exception-safe line.
        auto impl = std::make_unique<detail::SimulationInputHandleImpl>(*source.impl_);
        // Move ownership of the heap object to the handle with a custom deleter.
        impl_.reset(impl.release());
    }
    return *this;
}
/*! \endcond */

SimulationInput* SimulationInputHandle::get() const noexcept
{
    if (!impl_)
    {
        return nullptr;
    }
    return impl_->simulationInput();
}

SimulationInputHandle::operator bool() const
{
    if (impl_)
    {
        return bool(*impl_);
    }
    return false;
}

SimulationInputHandle makeSimulationInput(const LegacyMdrunOptions& options)
{
    // Note: it seems clear that we will want a SimulationInput to be linked to
    // a communications context (whether the SimulationContext or something higher level)
    // so that it can encapsulate the data locality details. Otherwise, we have
    // to choose whether to read the files everywhere or just to store the
    // filenames until a communications context is known.
    const auto* const tprFilename = ftp2fn(efTPR, options.filenames.size(), options.filenames.data());
    const auto* const cpiFilename = opt2fn("-cpi", options.filenames.size(), options.filenames.data());
    auto              simulationInput = std::make_unique<SimulationInput>(tprFilename, cpiFilename);
    auto impl = std::make_unique<detail::SimulationInputHandleImpl>(std::move(simulationInput));

    return SimulationInputHandle(std::move(impl));
}

} // end namespace gmx
