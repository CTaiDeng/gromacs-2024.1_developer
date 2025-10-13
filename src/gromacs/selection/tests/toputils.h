/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2013- The GROMACS Authors
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

/*! \internal \file
 * \brief
 * Helper routines for constructing topologies for tests.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_selection
 */
#ifndef GMX_SELECTION_TESTS_TOPUTILS_H
#define GMX_SELECTION_TESTS_TOPUTILS_H

#include <memory>
#include <vector>

struct gmx_mtop_t;
struct t_atoms;
struct t_trxframe;

namespace gmx
{

template<typename T>
class ArrayRef;

namespace test
{

/*! \internal
 * \brief Helps create and manage the lifetime of topology and
 * related data structures. */
class TopologyManager
{
public:
    TopologyManager();
    ~TopologyManager();

    //@{
    /*! \brief Prepare the memory within a trajectory frame needed
     * for test */
    void requestFrame();
    void requestVelocities();
    void requestForces();
    void initFrameIndices(const ArrayRef<const int>& index);
    //@}

    //! Load a topology from an input file relative to the source tree
    void loadTopology(const char* filename);

    //@{
    /*! \brief Functions for creating simplistic topologies
     *
     * These are easy to work with for some kinds of tests. */
    void initAtoms(int count);
    void initAtomTypes(const ArrayRef<const char* const>& types);
    void initUniformResidues(int residueSize);
    void initUniformMolecules(int moleculeSize);
    //@}

    //@{
    /*! \brief Functions for creating realistic topologies
     *
     * Real topologies aren't uniform, so we need to be able to
     * create custom topologies to test against.
     *
     * Ideally, we'd just be able to push new molecule types and
     * blocks, but the data structures are not mature enough for
     * that yet. The intended usage pattern is to initialize the
     * topology and declare the number of molecule types and
     * blocks, and then call the setter functions to fill in the
     * data structures. */
    void initTopology(int numMoleculeTypes, int numMoleculeBlocks);
    void setMoleculeType(int moleculeTypeIndex, ArrayRef<const int> residueSizes);
    void setMoleculeBlock(int moleculeBlockIndex, int moleculeTypeIndex, int numMoleculesToAdd);
    void finalizeTopology();
    //@}

    //@{
    //! Getters
    gmx_mtop_t* topology() { return mtop_.get(); }
    t_atoms&    atoms();
    t_trxframe* frame() { return frame_; }
    //@}

private:
    //! Topology
    std::unique_ptr<gmx_mtop_t> mtop_;
    //! Trajectory frame
    t_trxframe* frame_;
    //! Atom type names
    std::vector<char*> atomtypes_;
};

} // namespace test
} // namespace gmx

#endif
