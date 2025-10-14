/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2020- The GROMACS Authors
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

/*! \inpublicapi \file
 * \brief
 * Aggregates nblib public API headers
 *
 * \author Victor Holanda <victor.holanda@cscs.ch>
 * \author Joe Jordan <ejjordan@kth.se>
 * \author Prashanth Kanduri <kanduri@cscs.ch>
 * \author Sebastian Keller <keller@cscs.ch>
 * \author Artem Zhmurov <zhmurov@gmail.com>
 */
#ifndef NBLIB_HEADERS_H
#define NBLIB_HEADERS_H

#include "nblib/basicdefinitions.h"
#include "nblib/box.h"
#include "nblib/gmxcalculatorcpu.h"
#include "nblib/integrator.h"
#include "nblib/interactions.h"
#include "nblib/kerneloptions.h"
#include "nblib/listed_forces/bondtypes.h"
#include "nblib/listed_forces/calculator.h"
#include "nblib/listed_forces/definitions.h"
#include "nblib/molecules.h"
#include "nblib/nbnxmsetuphelpers.h"
#include "nblib/particlesequencer.h"
#include "nblib/particletype.h"
#include "nblib/simulationstate.h"
#include "nblib/topology.h"
#include "nblib/util/setup.h"
#include "nblib/util/traits.hpp"
#include "nblib/util/util.hpp"
#include "nblib/vector.h"

#endif // NBLIB_HEADERS_H
