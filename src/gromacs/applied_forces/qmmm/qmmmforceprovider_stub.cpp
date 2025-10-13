/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2021- The GROMACS Authors
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
 * Stub implementation of QMMMForceProvider
 * Compiled in case if CP2K is not linked
 *
 * \author Dmitry Morozov <dmitry.morozov@jyu.fi>
 * \author Christian Blau <blau@kth.se>
 * \ingroup module_applied_forces
 */

#include "gmxpre.h"

#include "gromacs/utility/exceptions.h"

#include "qmmmforceprovider.h"

namespace gmx
{


#ifdef __clang__
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wmissing-noreturn"
#endif

QMMMForceProvider::QMMMForceProvider(const QMMMParameters& parameters,
                                     const LocalAtomSet&   localQMAtomSet,
                                     const LocalAtomSet&   localMMAtomSet,
                                     PbcType               pbcType,
                                     const MDLogger&       logger) :
    parameters_(parameters),
    qmAtoms_(localQMAtomSet),
    mmAtoms_(localMMAtomSet),
    pbcType_(pbcType),
    logger_(logger),
    box_{ { 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0 } }
{
    GMX_THROW(
            InternalError("CP2K has not been linked into GROMACS, QMMM simulation is not "
                          "possible.\nPlease, reconfigure GROMACS with -DGMX_CP2K=ON\n"));
}

QMMMForceProvider::~QMMMForceProvider() {}

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
bool QMMMForceProvider::isQMAtom(Index /*globalAtomIndex*/)
{
    GMX_THROW(
            InternalError("CP2K has not been linked into GROMACS, QMMM simulation is not "
                          "possible.\nPlease, reconfigure GROMACS with -DGMX_CP2K=ON\n"));
}

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
void QMMMForceProvider::appendLog(const std::string& /*msg*/)
{
    GMX_THROW(
            InternalError("CP2K has not been linked into GROMACS, QMMM simulation is not "
                          "possible.\nPlease, reconfigure GROMACS with -DGMX_CP2K=ON\n"));
}

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
void QMMMForceProvider::initCP2KForceEnvironment(const t_commrec& /*cr*/)
{
    GMX_THROW(
            InternalError("CP2K has not been linked into GROMACS, QMMM simulation is not "
                          "possible.\nPlease, reconfigure GROMACS with -DGMX_CP2K=ON\n"));
}

void QMMMForceProvider::calculateForces(const ForceProviderInput& /*fInput*/, ForceProviderOutput* /*fOutput*/)
{
    GMX_THROW(
            InternalError("CP2K has not been linked into GROMACS, QMMM simulation is not "
                          "possible.\nPlease, reconfigure GROMACS with -DGMX_CP2K=ON\n"));
};

#ifdef __clang__
#    pragma clang diagnostic pop
#endif

} // namespace gmx
