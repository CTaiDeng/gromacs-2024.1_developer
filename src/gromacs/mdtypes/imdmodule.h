/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2017- The GROMACS Authors
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

/*! \libinternal \file
 * \brief
 * Declares gmx::IMDModule.
 *
 * See \ref page_mdmodules for an overview of this and associated interfaces.
 *
 * \inlibraryapi
 * \ingroup module_mdtypes
 */
#ifndef GMX_MDTYPES_IMDMODULE_H
#define GMX_MDTYPES_IMDMODULE_H


namespace gmx
{

class ForceProviders;
class IMDOutputProvider;
class IMdpOptionProvider;
struct MDModulesNotifiers;

/*! \libinternal \brief
 * Extension module for \Gromacs simulations.
 *
 * The methods that return other interfaces can in the future return null for
 * those interfaces that the module does not need to implement, but currently
 * the callers are not prepared to generically handle various cases.
 *
 * \inlibraryapi
 * \ingroup module_mdtypes
 */
class IMDModule
{
public:
    virtual ~IMDModule() {}

    //! Returns an interface for handling mdp input (and tpr I/O).
    virtual IMdpOptionProvider* mdpOptionProvider() = 0;
    //! Returns an interface for handling output files during simulation.
    virtual IMDOutputProvider* outputProvider() = 0;
    //! Initializes force providers from this module.
    virtual void initForceProviders(ForceProviders* forceProviders) = 0;
    //! Subscribe to simulation setup notifications
    virtual void subscribeToSimulationSetupNotifications(MDModulesNotifiers* notifiers) = 0;
    //! Subscribe to pre processing notifications
    virtual void subscribeToPreProcessingNotifications(MDModulesNotifiers* notifiers) = 0;
};

} // namespace gmx

#endif
