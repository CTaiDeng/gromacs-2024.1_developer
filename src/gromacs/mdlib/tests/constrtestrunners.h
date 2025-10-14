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

/*! \internal \file
 * \brief SHAKE and LINCS tests runners.
 *
 * Declares test runner class for constraints. The test runner abstract class is used
 * to unify the interfaces for different constraints methods, running on different
 * hardware.  This allows to run the same test on the same data using different
 * implementations of the parent class, that inherit its interfaces.
 *
 * \author Artem Zhmurov <zhmurov@gmail.com>
 * \ingroup module_mdlib
 */

#ifndef GMX_MDLIB_TESTS_CONSTRTESTRUNNERS_H
#define GMX_MDLIB_TESTS_CONSTRTESTRUNNERS_H

#include <gtest/gtest.h>

#include "testutils/test_device.h"

#include "constrtestdata.h"

/*
 * GPU version of constraints is only available with CUDA and SYCL.
 */
#define GPU_CONSTRAINTS_SUPPORTED (GMX_GPU_CUDA || GMX_GPU_SYCL)

struct t_pbc;

namespace gmx
{
namespace test
{

/* \brief Constraints test runner interface.
 *
 * Wraps the actual implementation of constraints algorithm into common interface.
 */
class IConstraintsTestRunner
{
public:
    //! Virtual destructor.
    virtual ~IConstraintsTestRunner() {}
    /*! \brief Abstract constraining function. Should be overriden.
     *
     * \param[in] testData             Test data structure.
     * \param[in] pbc                  Periodic boundary data.
     */
    virtual void applyConstraints(ConstraintsTestData* testData, t_pbc pbc) = 0;

    /*! \brief Get the name of the implementation.
     *
     * \return "<algorithm> on <device>", depending on the actual implementation used. E.g., "LINCS on #0: NVIDIA GeForce GTX 1660 SUPER".
     */
    virtual std::string name() = 0;
};

// Runner for the CPU implementation of SHAKE constraints algorithm.
class ShakeConstraintsRunner : public IConstraintsTestRunner
{
public:
    //! Default constructor.
    ShakeConstraintsRunner() {}
    /*! \brief Apply SHAKE constraints to the test data.
     *
     * \param[in] testData             Test data structure.
     * \param[in] pbc                  Periodic boundary data.
     */
    void applyConstraints(ConstraintsTestData* testData, t_pbc pbc) override;
    /*! \brief Get the name of the implementation.
     *
     * \return "SHAKE" string;
     */
    std::string name() override { return "SHAKE on CPU"; }
};

// Runner for the CPU implementation of LINCS constraints algorithm.
class LincsConstraintsRunner : public IConstraintsTestRunner
{
public:
    //! Default constructor.
    LincsConstraintsRunner() {}
    /*! \brief Apply LINCS constraints to the test data on the CPU.
     *
     * \param[in] testData             Test data structure.
     * \param[in] pbc                  Periodic boundary data.
     */
    void applyConstraints(ConstraintsTestData* testData, t_pbc pbc) override;
    /*! \brief Get the name of the implementation.
     *
     * \return "LINCS" string;
     */
    std::string name() override { return "LINCS on CPU"; }
};

// Runner for the GPU implementation of LINCS constraints algorithm.
class LincsDeviceConstraintsRunner : public IConstraintsTestRunner
{
public:
    /*! \brief Constructor. Keeps a copy of the hardware context.
     *
     * \param[in] testDevice The device hardware context to be used by the runner.
     */
    LincsDeviceConstraintsRunner(const TestDevice& testDevice) : testDevice_(testDevice) {}
    /*! \brief Apply LINCS constraints to the test data on the GPU.
     *
     * \param[in] testData             Test data structure.
     * \param[in] pbc                  Periodic boundary data.
     */
    void applyConstraints(ConstraintsTestData* testData, t_pbc pbc) override;
    /*! \brief Get the name of the implementation.
     *
     * \return "LINCS_GPU" string;
     */
    std::string name() override { return "LINCS on " + testDevice_.description(); }

private:
    //! Test device to be used in the runner.
    const TestDevice& testDevice_;
};

} // namespace test
} // namespace gmx

#endif // GMX_MDLIB_TESTS_CONSTRTESTRUNNERS_H
