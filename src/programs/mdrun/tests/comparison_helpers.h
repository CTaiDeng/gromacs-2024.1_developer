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

/*! \internal \file
 * \brief Declares types and functions common to comparing either
 * energies or trajectories produced by mdrun.
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \ingroup module_mdrun_integration_tests
 */
#ifndef GMX_PROGRAMS_MDRUN_TESTS_COMPARISON_HELPERS_H
#define GMX_PROGRAMS_MDRUN_TESTS_COMPARISON_HELPERS_H

#include <limits>

namespace gmx
{

namespace test
{

/*! \internal
 * \brief Named struct indicating the max number of frames to be compared */
struct MaxNumFrames
{
    //! Explicit constructor
    explicit MaxNumFrames(unsigned int maxFrame) : maxFrame_(maxFrame) {}

    //! Implicit conversion to int - struct can be used like underlying type
    operator unsigned int() const { return maxFrame_; }

    //! Return a MaxNumFrames that will try to compare all frames
    [[nodiscard]] static MaxNumFrames compareAllFrames()
    {
        return MaxNumFrames(std::numeric_limits<decltype(maxFrame_)>::max());
    }

private:
    //! Internal value
    const unsigned int maxFrame_;
};

} // namespace test
} // namespace gmx

#endif
