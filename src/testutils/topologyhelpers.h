/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2022- The GROMACS Authors
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

/*!\file
 * \internal
 * \brief
 * Helper functions to create test topologies.
 *
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 * \libinternal
 * \ingroup module_topology
 */

#ifndef GMX_TOPOLOGY_TESTHELPERS_H
#define GMX_TOPOLOGY_TESTHELPERS_H

struct gmx_mtop_t;

namespace gmx
{

namespace test
{

/*! \brief Adds water molecules with settles to topology
 *
 * Generates test topology with \p numWaters tip3p molecules.
 * Useful for testing topology methods that require a valid \p mtop.
 *
 * \param[in,out] mtop Handle to global topology.
 * \param[in] numWaters Number of water molecules that should
 *                      be added to the topology.
 */
void addNWaterMolecules(gmx_mtop_t* mtop, int numWaters);

} // namespace test

} // namespace gmx

#endif
