/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2015- The GROMACS Authors
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
 * Declares module creation function for applied electric fields
 *
 * \author David van der Spoel <david.vanderspoel@icm.uu.se>
 * \ingroup module_applied_forces
 * \inlibraryapi
 */
#ifndef GMX_APPLIED_FORCES_ELECTRICFIELD_H
#define GMX_APPLIED_FORCES_ELECTRICFIELD_H

#include <memory>

namespace gmx
{

class IMDModule;

/*! \brief
 * Creates a module for an external electric field.
 *
 * The returned class describes the time dependent electric field that can
 * be applied to all charges in a simulation. The field is described
 * by the following:
 *     E(t) = A cos(omega*(t-t0))*exp(-sqr(t-t0)/(2.0*sqr(sigma)));
 * If sigma = 0 there is no pulse and we have instead
 *     E(t) = A cos(omega*t)
 *
 * force is kJ mol^-1 nm^-1 = e * kJ mol^-1 nm^-1 / e
 *
 * WARNING:
 * There can be problems with the virial.
 * Since the field is not self-consistent this is unavoidable.
 * For neutral molecules the virial is correct within this approximation.
 * For neutral systems with many charged molecules the error is small.
 * But for systems with a net charge or a few charged molecules
 * the error can be significant when the field is high.
 * Solution: implement a self-consistent electric field into PME.
 */
std::unique_ptr<IMDModule> createElectricFieldModule();

} // namespace gmx

#endif
