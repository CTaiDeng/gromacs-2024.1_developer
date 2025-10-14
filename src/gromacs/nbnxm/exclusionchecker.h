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
 *
 * \brief This file declares functionality for checking whether
 * all exclusions are present in the pairlist
 *
 * \ingroup module_nbnxm
 */

#ifndef GMX_NBNXM_EXCLUSIONCHECKER_H
#define GMX_NBNXM_EXCLUSIONCHECKER_H

#include <memory>

struct gmx_mtop_t;
struct t_commrec;

namespace gmx
{
class ObservablesReducerBuilder;
}

/*! \internal \file
 * \brief Has responsibility for checking that the sum of the local number
 * of perturbed exclusions over the domains matches the global expected number.
 *
 * This uses the ObservablesReducer framework to check that the count
 * of perturbed exclusions with rlist assigned during pairlist generation
 * on each domain matches sums to the expected value.
 * Because this check is not urgent, the communication that it requires
 * is done at the next opportunity, rather than requiring extra communication.
 * If the check fails, a fatal error stops execution.
 */
class ExclusionChecker
{
public:
    /*! \brief Constructor
     * \param[in]    cr               Communication object
     * \param[in]    mtop             Global system topology
     * \param[in]    observablesReducerBuilder  Handle to builder for ObservablesReducer
     */
    ExclusionChecker(const t_commrec*                cr,
                     const gmx_mtop_t&               mtop,
                     gmx::ObservablesReducerBuilder* observablesReducerBuilder);
    //! Destructor
    ~ExclusionChecker();
    //! Move constructor
    ExclusionChecker(ExclusionChecker&& other) noexcept;
    //! Move assignment
    ExclusionChecker& operator=(ExclusionChecker&& other) noexcept;

    /*! \brief Set that the exclusion count should be checked via
     * observables reduction whenever that reduction is required by
     * another module. In case of a single domain the check is performed
     * directly instead.
     */
    void scheduleCheckOfExclusions(int numPerturbedExclusionsToReduce);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

#endif
