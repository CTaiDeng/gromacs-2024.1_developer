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
 * Declares the HistogramSize class.
 *
 * The data members of this class keep track of global size and update related
 * properties of the bias histogram and the evolution of the histogram size.
 * Initially histogramSize_ (and thus the convergence rate) is controlled
 * heuristically to get good initial estimates,  i.e. increase the robustness
 * of the method.
 *
 * \author Viveca Lindahl
 * \author Berk Hess <hess@kth.se>
 * \ingroup module_awh
 */

#ifndef GMX_AWH_HISTOGRAMSIZE_H
#define GMX_AWH_HISTOGRAMSIZE_H

#include <cstdio>

#include <vector>

#include "gromacs/math/vectypes.h"

namespace gmx
{

template<typename>
class ArrayRef;
struct AwhBiasStateHistory;
class AwhBiasParams;
class BiasParams;
class PointState;

/*! \internal
 * \brief Tracks global size related properties of the bias histogram.
 *
 * Tracks the number of updates and the histogram size.
 * Also keep track of the stage (initial/final of the AWH method
 * and printing warnings about covering.
 *
 * \note Histogram sizes are floating-point values, since the histogram uses weighted
 *        entries and we can assign a floating-point scaling factor when changing it.
 */
class HistogramSize
{
public:
    /*! \brief Constructor.
     *
     * \param[in] awhBiasParams         The Bias parameters from inputrec.
     * \param[in] histogramSizeInitial  The initial histogram size.
     */
    HistogramSize(const AwhBiasParams& awhBiasParams, double histogramSizeInitial);

private:
    /*! \brief
     * Returns the new size of the reference weight histogram in the initial stage.
     *
     * This function also takes care resetting the histogram used for covering checks
     * and for exiting the initial stage.
     *
     * \param[in]     params             The bias parameters.
     * \param[in]     t                  Time.
     * \param[in]     detectedCovering   True if we detected that the sampling interval has been
     *                                   sufficiently covered.
     * \param[in,out] weightsumCovering  The weight sum for checking covering.
     * \param[in,out] fplog              Log file.
     * \returns the new histogram size.
     */
    double newHistogramSizeInitialStage(const BiasParams& params,
                                        double            t,
                                        bool              detectedCovering,
                                        ArrayRef<double>  weightsumCovering,
                                        FILE*             fplog);

public:
    /*! \brief
     * Return the new reference weight histogram size for the current update.
     *
     * This function also takes care of checking for covering in the initial stage.
     *
     * \param[in]     params             The bias parameters.
     * \param[in]     t                  Time.
     * \param[in]     covered            True if the sampling interval has been covered enough.
     * \param[in]     pointStates        The state of the grid points.
     * \param[in,out] weightsumCovering  The weight sum for checking covering.
     * \param[in,out] fplog              Log file.
     * \returns the new histogram size.
     */
    double newHistogramSize(const BiasParams&          params,
                            double                     t,
                            bool                       covered,
                            ArrayRef<const PointState> pointStates,
                            ArrayRef<double>           weightsumCovering,
                            FILE*                      fplog);

    /*! \brief Restores the histogram size from history.
     *
     * \param[in] stateHistory  The AWH bias state history.
     */
    void restoreFromHistory(const AwhBiasStateHistory& stateHistory);

    /*! \brief Store the histogram size state in a history struct.
     *
     * \param[in,out] stateHistory  The AWH bias state history.
     */
    void storeState(AwhBiasStateHistory* stateHistory) const;

    /*! \brief Returns the number of updates since the start of the simulation.
     */
    int numUpdates() const { return numUpdates_; }

    /*! \brief Increments the number of updates by 1.
     */
    void incrementNumUpdates() { numUpdates_ += 1; }

    /*! \brief Returns the histogram size.
     */
    double histogramSize() const { return histogramSize_; }

    /*! \brief Sets the histogram size.
     *
     * \param[in] histogramSize                 The new histogram size.
     * \param[in] weightHistogramScalingFactor  The factor to scale the weight by.
     */
    void setHistogramSize(double histogramSize, double weightHistogramScalingFactor);

    /*! \brief Returns true if we are in the initial stage of the AWH method.
     */
    bool inInitialStage() const { return inInitialStage_; }

    /*! \brief Returns The log of the current sample weight, scaled because of the histogram rescaling.
     */
    double logScaledSampleWeight() const { return logScaledSampleWeight_; }

private:
    int64_t numUpdates_; /**< The number of updates performed since the start of the simulation. */

    /* The histogram size sets the update size and so controls the convergence rate of the free energy and bias. */
    double histogramSize_; /**< Size of reference weight histogram. */

    /* Values that control the evolution of the histogram size. */
    bool   inInitialStage_;       /**< True if in the initial stage. */
    double growthFactor_;         /**< The growth factor for the initial stage */
    bool   equilibrateHistogram_; /**< True if samples are kept from accumulating until the sampled distribution is close enough to the target. */
    double logScaledSampleWeight_; /**< The log of the current sample weight, scaled because of the histogram rescaling. */
    double maxLogScaledSampleWeight_; /**< Maximum sample weight obtained for previous (smaller) histogram sizes. */

    /* Bool to avoid printing multiple, not so useful, messages to log */
    bool havePrintedAboutCovering_; /**< True if we have printed about covering to the log while equilibrateHistogram==true */
};

} // namespace gmx

#endif /* GMX_AWH_HISTOGRAMSIZE_H */
