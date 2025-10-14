/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2012- The GROMACS Authors
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
 * Declares gmx::AnalysisDataParallelOptions.
 *
 * \if internal
 * Implementation of this class is currently in datastorage.cpp.
 * \endif
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \inlibraryapi
 * \ingroup module_analysisdata
 */
#ifndef GMX_ANALYSISDATA_PARALLELOPTIONS_H
#define GMX_ANALYSISDATA_PARALLELOPTIONS_H

namespace gmx
{

/*! \libinternal \brief
 * Parallelization options for analysis data objects.
 *
 * Methods in this class do not throw.
 *
 * \inlibraryapi
 * \ingroup module_analysisdata
 */
class AnalysisDataParallelOptions
{
public:
    //! Constructs options for serial execution.
    AnalysisDataParallelOptions();
    /*! \brief
     * Constructs options for parallel execution with given number of
     * concurrent frames.
     *
     * \param[in] parallelizationFactor
     *      Number of frames that may be constructed concurrently.
     *      Must be >= 1.
     */
    explicit AnalysisDataParallelOptions(int parallelizationFactor);

    //! Returns the number of frames that may be constructed concurrently.
    int parallelizationFactor() const { return parallelizationFactor_; }

private:
    int parallelizationFactor_;
};

} // namespace gmx

#endif
