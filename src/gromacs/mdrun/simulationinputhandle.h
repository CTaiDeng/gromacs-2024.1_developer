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

/*! \file
 * \brief Public interface for SimulationInput facilities.
 *
 * \author M. Eric Irrgang <ericirrgang@gmail.com>
 * \ingroup module_mdrun
 * \inpublicapi
 */

#ifndef GMX_MDRUN_SIMULATIONINPUTHANDLE_H
#define GMX_MDRUN_SIMULATIONINPUTHANDLE_H

#include <memory>

namespace gmx
{

// Forward declarations for types from other modules that are opaque to the public API.
class LegacyMdrunOptions;

/*!
 * \brief Prescription for molecular simulation.
 *
 * Represent the complete and unique information needed to generate a simulation
 * trajectory segment. SimulationInput objects are opaque to the public API.
 * Ownership can be managed with SimulationInputHolder objects. Clients can
 * acquire owning references to SimulationInput objects (as SimulationInputHolder)
 * through makeSimulationInput() or from other SimulationInputHolders.
 *
 * A SimulationInput object represents an immutable source of data, and is safe
 * to share. A SimulationInput object may have internal state to support
 * performance optimizations when shared by multiple SimulationInputHolders.
 * The SimulationInput is guaranteed to live at least as long as any associated
 * SimulationInputHolders. The API does not specify whether it may persist
 * longer internally or be reused for later equivalent requests.
 *
 * \see SimulationInputHandle
 * \see makeSimulationInput()
 *
 * \internal
 * SimulationInput has no public interface yet, but we need a forward declaration for the
 * library symbol. Library interface provided through simulationinput.h
 * See also https://gitlab.com/gromacs/gromacs/-/issues/3379 for design and development road map.
 */
class SimulationInput;

/*! \cond internal
 * Client software developers do not interact directly with the contents of gmx::detail,
 * but some declarations and definitions are necessary in the public headers, such as
 * forward declarations of implementation classes and definitions of custom deleters.
 */
namespace detail
{
/*!
 * \brief Private implementation class;
 */
class SimulationInputHandleImpl;

/*!
 * \brief Explicit deleter details for SimulationInputHolderImpl.
 *
 * SimulationInputHolderImpl objects are created by the GROMACS library, but
 * are destroyed when the SimulationInputHolder goes out of scope in the client
 * code, which may be linked to a different allocator.
 * We want to make sure that objects are allocated and deallocated with the same
 * allocator, so we avoid the default deleter in unique_ptrs and compile allocation
 * and deallocation code in the same translation unit.
 *
 * Note that this does not solve potential ABI incompatibilities between the
 * unique_ptr implementations themselves, but we need to consider ABI
 * compatibility goals and challenges as well as supported use cases and
 * ownership semantics.
 */
struct SimulationInputHandleImplDeleter
{
    /*! \cond */
    SimulationInputHandleImplDeleter();
    SimulationInputHandleImplDeleter(const SimulationInputHandleImplDeleter& deleter) noexcept;
    SimulationInputHandleImplDeleter(SimulationInputHandleImplDeleter&& deleter) noexcept;
    SimulationInputHandleImplDeleter& operator=(const SimulationInputHandleImplDeleter& deleter) noexcept;
    SimulationInputHandleImplDeleter& operator=(SimulationInputHandleImplDeleter&& deleter) noexcept;
    void                              operator()(SimulationInputHandleImpl* impl) const;
    /*! \endcond */
};
} // end namespace detail
/*! \endcond internal */

/*!
 * \brief Owning handle to a SimulationInput object.
 *
 * SimulationInput objects are logically immutable, so ownership may be shared
 * by multiple SimulationInputHolders.
 *
 * Acquire a SimulationInputHolder with makeSimulationInput() and pass to (e.g.)
 * gmx::MdrunnerBuilder::addInput()
 *
 * SimulationInput has no public API yet.
 * \see https://gitlab.com/gromacs/gromacs/-/issues/3379
 */
class SimulationInputHandle
{
public:
    /*! \cond internal */
    SimulationInputHandle();
    ~SimulationInputHandle();

    SimulationInputHandle(const SimulationInputHandle& source);
    SimulationInputHandle(SimulationInputHandle&&) noexcept = default;

    SimulationInputHandle& operator=(const SimulationInputHandle& rhs);
    SimulationInputHandle& operator=(SimulationInputHandle&&) noexcept = default;

    /*!
     * \brief Take ownership of private implementation object to produce a new public holder.
     */
    explicit SimulationInputHandle(std::unique_ptr<detail::SimulationInputHandleImpl> impl);
    /*! \endcond */

    /*!
     * \brief Access opaque SimulationInput pointer.
     *
     * \return Borrowed access to the SimulationInput, if present.
     */
    [[nodiscard]] SimulationInput* get() const noexcept;

    /*!
     * \brief Boolean context returns true if an input is held, else false.
     */
    operator bool() const;

private:
    std::unique_ptr<detail::SimulationInputHandleImpl, detail::SimulationInputHandleImplDeleter> impl_;
};

/*! \brief Direct the construction of a SimulationInput.
 *
 * \warning Creation methods for SimulationInput resources are under rapid development.
 * Reference https://gitlab.com/gromacs/gromacs/-/issues/3652
 *
 * \param options library-internal container holding partially processed user input.
 *
 * \ingroup module_mdrun
 *
 * \internal
 * This isn't really a public API function until its arguments are obtainable
 * through the public API.
 *
 * Design notes: SimulationInput creation will warrant a builder protocol, and
 * this helper can evolve into a director to apply the contents of LegacyMdrunOptions,
 * while such an operation is still relevant.
 *
 * Example:
 *     // After preparing a LegacyMdrunOptions and calling handleRestart()...
 *     SimulationInputBuilder builder;
 *     auto simulationInputHandle = makeSimulationInput(options, &builder);
 *
 *     // In addition to MdrunnerBuilder::addFiles(),
 *     mdrunnerBuilder.addInput(simulationInputHandle.get());
 *
 */
SimulationInputHandle makeSimulationInput(const LegacyMdrunOptions& options);

} // end namespace gmx

#endif // GMX_MDRUN_SIMULATIONINPUTHANDLE_H
