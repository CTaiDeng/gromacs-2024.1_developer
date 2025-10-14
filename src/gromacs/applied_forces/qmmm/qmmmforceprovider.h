/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2021- The GROMACS Authors
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

/*! \internal \file
 * \brief
 * Declares force provider for QMMM
 *
 * \author Dmitry Morozov <dmitry.morozov@jyu.fi>
 * \author Christian Blau <blau@kth.se>
 * \ingroup module_applied_forces
 */
#ifndef GMX_APPLIED_FORCES_QMMMFORCEPROVIDER_H
#define GMX_APPLIED_FORCES_QMMMFORCEPROVIDER_H

#include "gromacs/domdec/localatomset.h"
#include "gromacs/mdtypes/forceoutput.h"
#include "gromacs/mdtypes/iforceprovider.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/utility/classhelpers.h"
#include "gromacs/utility/logger.h"

#include "qmmmtypes.h"

namespace gmx
{

#ifdef __clang__
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wunused-private-field"
#endif

//! Type for CP2K force environment handle
typedef int force_env_t;

/*! \internal \brief
 * Implements IForceProvider for QM/MM.
 */
class QMMMForceProvider final : public IForceProvider
{
public:
    QMMMForceProvider(const QMMMParameters& parameters,
                      const LocalAtomSet&   localQMAtomSet,
                      const LocalAtomSet&   localMMAtomSet,
                      PbcType               pbcType,
                      const MDLogger&       logger);

    //! Destruct force provider for QMMM and finalize libcp2k
    ~QMMMForceProvider();

    /*!\brief Calculate forces of QMMM.
     * \param[in] fInput input for force provider
     * \param[out] fOutput output for force provider
     */
    void calculateForces(const ForceProviderInput& fInput, ForceProviderOutput* fOutput) override;

private:
    //! Write message to the log
    void appendLog(const std::string& msg);

    /*!\brief Check if atom belongs to the global index of qmAtoms_
     * \param[in] globalAtomIndex global index of the atom to check
     */
    bool isQMAtom(Index globalAtomIndex);

    /*!\brief Initialization of QM program.
     * \param[in] cr connection record structure
     */
    void initCP2KForceEnvironment(const t_commrec& cr);

    const QMMMParameters& parameters_;
    const LocalAtomSet&   qmAtoms_;
    const LocalAtomSet&   mmAtoms_;
    const PbcType         pbcType_;
    const MDLogger&       logger_;

    //! Internal copy of PBC box
    matrix box_;

    //! Flag wether initCP2KForceEnvironment() has been called already
    bool isCp2kLibraryInitialized_ = false;

    //! CP2K force environment handle
    force_env_t force_env_ = -1;
};

#ifdef __clang__
#    pragma clang diagnostic pop
#endif

} // namespace gmx

#endif // GMX_APPLIED_FORCES_QMMMFORCEPROVIDER_H
