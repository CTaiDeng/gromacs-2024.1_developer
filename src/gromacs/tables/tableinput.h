/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2016- The GROMACS Authors
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
 * Declares structures for analytical or numerical input data to construct tables
 *
 * \inlibraryapi
 * \author Erik Lindahl <erik.lindahl@gmail.com>
 * \ingroup module_tables
 */

#ifndef GMX_TABLES_TABLEINPUT_H
#define GMX_TABLES_TABLEINPUT_H

#include <functional>
#include <vector>

#include "gromacs/utility/arrayref.h"

namespace gmx
{

/*! \libinternal \brief Specification for analytical table function (name, function, derivative)
 */
struct AnalyticalSplineTableInput
{
    //NOLINTNEXTLINE(google-runtime-member-string-references)
    const std::string&            desc;       //!< \libinternal Brief description of function
    std::function<double(double)> function;   //!< \libinternal Analytical form of function
    std::function<double(double)> derivative; //!< \libinternal Analytical derivative
};

/*! \libinternal \brief Specification for vector table function (name, function, derivative, spacing)
 */
struct NumericalSplineTableInput
{
    //NOLINTNEXTLINE(google-runtime-member-string-references)
    const std::string&     desc;       //!< \libinternal Brief description of function
    ArrayRef<const double> function;   //!< \libinternal Vector with function values
    ArrayRef<const double> derivative; //!< \libinternal Vector with derivative values
    double                 spacing;    //!< \libinternal Distance between data points
};


} // namespace gmx


#endif // GMX_TABLES_TABLEINPUT_H
