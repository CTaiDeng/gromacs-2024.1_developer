/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 1991- The GROMACS Authors
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

#ifndef GMX_TOPOLOGY_TOPOLOGY_ENUMS_H
#define GMX_TOPOLOGY_TOPOLOGY_ENUMS_H

enum class SimulationAtomGroupType : int
{
    TemperatureCoupling,
    EnergyOutput,
    Acceleration,
    Freeze,
    User1,
    User2,
    MassCenterVelocityRemoval,
    CompressedPositionOutput,
    OrientationRestraintsFit,
    QuantumMechanics,
    Count
};

//! Short strings used for describing atom groups in log and energy files
const char* shortName(SimulationAtomGroupType type);

/* The particle type */
enum class ParticleType : int
{
    Atom,
    Nucleus,
    Shell,
    Bond,
    VSite,
    Count
};

/* The particle type names */
const char* enumValueToString(ParticleType enumValue);

/* Enumerated type for pdb records. The other entries are ignored
 * when reading a pdb file
 */
enum class PdbRecordType : int
{
    Atom,
    Hetatm,
    Anisou,
    Cryst1,
    Compound,
    Model,
    EndModel,
    Ter,
    Header,
    Title,
    Remark,
    Conect,
    Count
};

const char* enumValueToString(PdbRecordType enumValue);

#endif
