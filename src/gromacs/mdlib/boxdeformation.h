/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2018- The GROMACS Authors
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
 * \brief Declares interface to box deformation code.
 *
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \ingroup module_mdlib
 * \inlibraryapi
 */

#ifndef GMX_MDRUN_BOXDEFORMATION_H
#define GMX_MDRUN_BOXDEFORMATION_H

#include <memory>

#include "gromacs/math/matrix.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/classhelpers.h"
#include "gromacs/utility/gmxmpi.h"

struct t_inputrec;
enum class DDRole;
enum class NumRanks;

namespace gmx
{

template<typename>
class ArrayRef;
class BoxDeformation
{
public:
    //! Trivial constructor.
    BoxDeformation(double           timeStep,
                   int64_t          initialStep,
                   const Matrix3x3& deformationTensor,
                   const Matrix3x3& referenceBox);

    //! Deform \c box at this \c step;
    void apply(Matrix3x3* box, int64_t step);

private:
    //! The integrator time step.
    double timeStep_;
    //! The initial step number (from the .tpr, which permits checkpointing to work correctly).
    int64_t initialStep_;
    //! Non-zero elements provide a scaling factor for deformation in that box dimension.
    Matrix3x3 deformationTensor_;
    //! The initial box, ie from the .tpr file.
    Matrix3x3 referenceBox_;
};

/*! \brief Factory function for box deformation module.
 *
 * If the \c inputrec specifies the use of box deformation during the
 * update phase, communicates the \c initialBox from SIMMAIN to
 * other ranks, and constructs and returns an object to manage that
 * update.
 *
 * \throws NotImplementedError if the \c inputrec specifies an
 * unsupported combination.
 */
std::unique_ptr<BoxDeformation> buildBoxDeformation(const Matrix3x3&  initialBox,
                                                    DDRole            ddRole,
                                                    NumRanks          numRanks,
                                                    MPI_Comm          communicator,
                                                    const t_inputrec& inputrec);

/*! \brief Set a matrix for computing the flow velocity at coordinates
 *
 * Used with continuous box deformation for calculating the flow profile.
 * Sets a matrix which can be used to multiply with coordinates to obtain
 * the flow velocity at that coordinate.
 *
 * \param[in]  boxDeformationVelocity  The velocity of the box in nm/ps
 * \param[in]  box                     The box in nm
 * \param[out] flowMatrix              The deformation rate in ps^-1
 */
void setBoxDeformationFlowMatrix(const matrix boxDeformationVelocity, const matrix box, matrix flowMatrix);

} // namespace gmx

#endif
