/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2018- The GROMACS Authors
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

#ifndef GROMACS_WORKFLOW_IMPL_H
#define GROMACS_WORKFLOW_IMPL_H

/*! \internal \file
 * \brief Implementation details for Workflow infrastructure.
 *
 * \author M. Eric Irrgang <ericirrgang@gmail.com>
 * \ingroup gmxapi
 */

#include <memory>
#include <string>

#include "gmxapi/exceptions.h"

// Local module internal headers.
#include "workflow.h"

namespace gmxapi
{

class WorkflowKeyError : public BasicException<WorkflowKeyError>
{
public:
    using BasicException::BasicException;
};

/*!
 * \brief Work graph node for MD simulation.
 */
class MDNodeSpecification : public NodeSpecification
{
public:
    //! Uses parameter type of base class.
    using NodeSpecification::paramsType;

    /*!
     * \brief Simulation node from file input
     *
     * \param filename TPR input filename.
     */
    explicit MDNodeSpecification(const std::string& filename);

    /*
     * \brief Implement NodeSpecification::clone()
     *
     * \returns a node to launch a simulation from the same input as this
     *
     * Returns nullptr if clone is not possible.
     */
    std::unique_ptr<NodeSpecification> clone() override;

    /*! \brief Implement NodeSpecification::params()
     *
     * \return Copy of internal params value.
     */
    paramsType params() const noexcept override;

private:
    //! The TPR input filename, set during construction
    paramsType tprfilename_;
};


} // end namespace gmxapi

#endif // GROMACS_WORKFLOW_IMPL_H
