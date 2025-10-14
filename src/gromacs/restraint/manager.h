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

#ifndef GROMACS_RESTRAINT_MANAGER_H
#define GROMACS_RESTRAINT_MANAGER_H

/*! \libinternal \file
 * \brief Declare the Manager for restraint potentials.
 *
 * \author M. Eric Irrgang <ericirrgang@gmail.com>
 *
 * \inlibraryapi
 * \ingroup module_restraint
 */

#include <memory>
#include <mutex>
#include <string>

#include "gromacs/restraint/restraintpotential.h"
#include "gromacs/utility/basedefinitions.h"

struct t_commrec;
struct t_mdatoms;
struct pull_t;

namespace gmx
{

/*! \libinternal \ingroup module_restraint
 * \brief Manage the Restraint potentials available for Molecular Dynamics.
 *
 * A simulation runner owns one manager resource to hold restraint objects used
 * in the simulation. In the case of thread MPI simulations, multiple runner
 * instances will have handles to the same underlying resource. With further
 * factoring of the mdrun call stack, this facility can be combined with others
 * into a simulation context object from which simulation code can retrieve
 * support code for a user-configured simulation.
 *
 * Calling code provides the manager with a means to access the various required input data
 * to be used when restraints are computed.
 *
 * \todo This should be generalized as work description and factory functions in Context.
 */
class RestraintManager final
{
public:
    //! Create new restraint manager resources with empty set of restraints.
    RestraintManager();

    ~RestraintManager();

    /*!
     * \brief Client code can access the shared resource by copying or moving a handle.
     * \{
     */
    RestraintManager(const RestraintManager& /* unused */) = default;
    RestraintManager& operator=(const RestraintManager& /* unused */) = default;
    RestraintManager(RestraintManager&&) noexcept                     = default;
    RestraintManager& operator=(RestraintManager&& /* unused */) noexcept = default;
    /*! \} */

    /*!
     * \brief Clear registered restraints and reset the manager.
     */
    void clear() noexcept;

    /*!
     * \brief Get the number of currently managed restraints.
     *
     * \return number of restraints.
     *
     * \internal
     * Only considers the IRestraintPotential objects
     */
    unsigned long countRestraints() noexcept;

    /*! \brief Obtain the ability to create a restraint MDModule
     *
     * Though the name is reminiscent of the evolving idea of a work specification, the
     * Spec here is just a list of restraint modules.
     *
     * \param restraint shared ownership of a restraint potential interface.
     * \param name key by which to reference the restraint.
     */
    void addToSpec(std::shared_ptr<gmx::IRestraintPotential> restraint, const std::string& name);

    /*!
     * \brief Get a copy of the current set of restraints to be applied.
     *
     * This function is to be used when launching a simulation to get the
     * restraint handles to bind, so it is not performance sensitive. A new
     * vector is returned with each call because it is unspecified whether
     * the set of handles point to the same objects on all threads or between
     * calls to getRestraints.
     *
     * \return a copy of the list of restraint potentials.
     */
    std::vector<std::shared_ptr<IRestraintPotential>> getRestraints() const;

private:
    class Impl;
    //! Ownership of the shared reference to the global manager.
    std::shared_ptr<Impl> instance_;
};

} // end namespace gmx

#endif // GROMACS_RESTRAINT_MANAGER_H
