/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2015- The GROMACS Authors
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
 *
 * \brief
 * Declares the CorrelationGrid class to collect correlation statistics on a grid, using several block lengths.
 *
 * \author Viveca Lindahl
 * \author Berk Hess <hess@kth.se>
 * \ingroup module_awh
 */

#ifndef GMX_AWH_CORRELATIONGRID_H
#define GMX_AWH_CORRELATIONGRID_H

#include <cstddef>

#include <vector>

#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/gmxassert.h"

#include "correlationtensor.h"

namespace gmx
{

template<typename>
class ArrayRef;
struct CorrelationGridHistory;

/*! \internal
 * \brief BiasGrid of local correlation tensors.
 *
 * This class provides the means for a bias to interaction with the grid
 * of correlation tensors. The grid should have the same number of points
 * and the same dimensionality as the bias grid.
 */
class CorrelationGrid
{
public:
    //! Enum that sets how we measure block length.
    enum class BlockLengthMeasure
    {
        Time,  //!< Measure block length in time.
        Weight //!< Measure block length in sampled weight.
    };

    /*! \brief Constructor.
     *
     * \param[in] numPoints           Number of points in the grid.
     * \param[in] numDims             Number of dimensions of the grid.
     * \param[in] blockLengthInit     Initial length of the blocks used for block averaging.
     * \param[in] blockLengthMeasure  Sets how we measure block length.
     * \param[in] dtSample            Time step for sampling correlations.
     */
    CorrelationGrid(int                numPoints,
                    int                numDims,
                    double             blockLengthInit,
                    BlockLengthMeasure blockLengthMeasure,
                    double             dtSample);

    /*! \brief Adds a weighted data vector to one point in the correlation grid.
     *
     * \param[in] pointIndex  Index of the point to add data to.
     * \param[in] weight      Weight to assign to the data.
     * \param[in] data        One data point for each grid dimension.
     * \param[in] t           The time when the data was sampled.
     */
    void addData(int pointIndex, double weight, gmx::ArrayRef<const double> data, double t)
    {
        tensors_[pointIndex].addData(weight, data, blockLengthMeasure_ == BlockLengthMeasure::Weight, t);
    }

    /*! \brief Restores the correlation grid state from the correlation grid history.
     *
     * The setup in the history should match that of this simulation.
     * If this is not the case, an exception is thrown.
     *
     * \param[in] correlationGridHistory  The correlation grid state history.
     */
    void restoreStateFromHistory(const CorrelationGridHistory& correlationGridHistory);

    /*! \brief Returns the number of elements in the tensor: dim*(dim+1)/2.
     */
    int tensorSize() const
    {
        GMX_RELEASE_ASSERT(!tensors_.empty(), "Should only call tensorSize on a valid grid");

        return tensors_[0].blockDataList()[0].correlationIntegral().size();
    }

    /*! \brief Returns the size of the block data list.
     */
    int blockDataListSize() const
    {
        GMX_RELEASE_ASSERT(!tensors_.empty(), "Should only call tensorSize on a valid grid");

        return tensors_[0].blockDataList().size();
    }

    /*! \brief Get a const reference to the correlation grid data.
     */
    const std::vector<CorrelationTensor>& tensors() const { return tensors_; }

    /* Right now the below functions are only used for an initial log printing. */

    /*! \brief Get the current blocklength.
     */
    double getBlockLength() const;

    /*! \brief Get the current number of blocks.
     *
     * If we have a finite block span we have a constant number of blocks,
     * otherwise we are always adding more blocks (and we don't keep
     * track of the number), so we return -1.
     */
    int getNumBlocks() const;

    const double             dtSample_;           /**< Time in between samples. */
    const BlockLengthMeasure blockLengthMeasure_; /**< The measure for the block length. */
private:
    std::vector<CorrelationTensor> tensors_; /**< Correlation tensor grid */
};

} // namespace gmx

#endif /* GMX_AWH_CORRELATIONGRID_H */
