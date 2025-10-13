/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2014- The GROMACS Authors
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
 * Declares helper class for autocorrelation tests
 *
 * \author Anders G&auml;rden&auml;s <anders.gardenas@gmail.com>
 * \ingroup module_correlationfunctions
 */
#ifndef GMX_CORRELATIONDATASET_H
#define GMX_CORRELATIONDATASET_H

#include <memory>
#include <string>
#include <vector>

#include "gromacs/utility/classhelpers.h"
#include "gromacs/utility/real.h"

class CorrelationDataSet
{
    double** tempValues_;

    int    nrLines_;
    int    nrColumns_;
    double startTime_;
    double endTime_;
    double dt_;

public:
    /*! \brief
     * Constructor
     * \param[in] fileName containing function to test. *.xvg
     */
    explicit CorrelationDataSet(const std::string& fileName);

    /*! \brief
     * Return a value at an index
     * \param[in] set the set number
     * \param[in] t the time index of the value
     */
    real getValue(int set, int t) const;

    /*! \brief
     * Return the nummber of columns
     */
    int getNrColumns() const { return nrColumns_; }

    /*! \brief
     * Return the nummber of Lines
     */
    int getNrLines() const { return nrLines_; }

    /*! \brief
     * Return the time witch the function starts at
     */
    real getStartTime() const { return startTime_; }

    /*! \brief
     * Return the time the function ends at
     */
    real getEndTime() const { return endTime_; }

    /*! \brief
     * return delta time
     */
    real getDt() const { return dt_; }

    /*! \brief
     * Destructor
     */
    ~CorrelationDataSet();

private:
    //! This class should not be copyable or assignable
    GMX_DISALLOW_COPY_AND_ASSIGN(CorrelationDataSet);
};

#endif
