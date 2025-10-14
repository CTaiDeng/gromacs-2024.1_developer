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
 * \brief
 * Contains classes and methods related to use of MuParser in pulling
 *
 * \author Oliver Fleetwood <oliver.fleetwood@gmail.com>
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 * \author Joe Jordan <ejjordan@kth.se>
 * \author Berk Hess <hess@kth.se>
 *
 */
#ifndef GMX_PULL_PULLCOORDEXPRESSIONPARSER_H
#define GMX_PULL_PULLCOORDEXPRESSIONPARSER_H

#include "config.h"

#include <memory>
#include <string>
#include <vector>

#if HAVE_MUPARSER
#    include <muParser.h>
#else
namespace mu
{
//! Defines a dummy Parser type to reduce use of the preprocessor.
using Parser = std::false_type;
} // namespace mu
#endif

struct pull_coord_work_t;

namespace gmx
{
template<typename>
class ArrayRef;

/*! \brief Class with a mathematical expression and parser.
 * \internal
 *
 * The class handles parser instantiation from an mathematical expression, e.g. 'x1*x2',
 * and evaluates the expression given the variables' numerical values.
 *
 * Note that for performance reasons you should not create a new PullCoordExpressionParser
 * for every evaluation.
 * Instead, instantiate one PullCoordExpressionParser per expression,
 * then update the variables before the next evaluation.
 *
 */
class PullCoordExpressionParser
{
public:
    //! Constructor which takes a mathematical expression and the number of variables as arguments
    PullCoordExpressionParser(const std::string& expression, int numVariables, bool allowTimeAsVariable);

    //! Evaluates the expression with the numerical values passed in \p variables.
    double evaluate(ArrayRef<const double> variables);

private:
    /*! \brief The mathematical expression, e.g. 'x1*x2' */
    std::string expression_;

    /*! \brief A vector containing the numerical values of the variables before parser evaluation.
     *
     * muParser compiles the expression to bytecode, then binds to the memory address
     * of these vector elements, making the evaluations fast and memory efficient.
     * */
    std::vector<double> variableValues_;

    /*! \brief The parser_ which compiles and evaluates the mathematical expression */
    std::unique_ptr<mu::Parser> parser_;
};

} // namespace gmx

#endif // GMX_PULL_PULLCOORDEXPRESSIONPARSER_H
