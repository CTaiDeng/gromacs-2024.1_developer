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
 * \brief
 * Tests for functionality of the QMMM module.
 *
 * \author Dmitry Morozov <dmitry.morozov@jyu.fi>
 * \author Christian Blau <blau@kth.se>
 * \ingroup module_applied_forces
 */
#include "gmxpre.h"

#include "gromacs/applied_forces/qmmm/qmmm.h"

#include <gtest/gtest.h>

#include "gromacs/gmxlib/network.h"
#include "gromacs/math/paddedvector.h"
#include "gromacs/math/vec.h"
#include "gromacs/mdrunutility/mdmodulesnotifiers.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/mdtypes/forceoutput.h"
#include "gromacs/mdtypes/iforceprovider.h"
#include "gromacs/mdtypes/imdmodule.h"
#include "gromacs/mdtypes/imdpoptionprovider.h"
#include "gromacs/mdtypes/mdatom.h"
#include "gromacs/options/options.h"
#include "gromacs/options/treesupport.h"
#include "gromacs/utility/keyvaluetreebuilder.h"
#include "gromacs/utility/keyvaluetreetransform.h"
#include "gromacs/utility/real.h"
#include "gromacs/utility/smalloc.h"
#include "gromacs/utility/stringcompare.h"

#include "testutils/testasserts.h"
#include "testutils/testfilemanager.h"
#include "testutils/testmatchers.h"

namespace gmx
{

namespace
{

class QMMMTest : public ::testing::Test
{
public:
    void addMdpOptionQMMMActive()
    {
        mdpValueBuilder_.rootObject().addValue("qmmm-cp2k-active", std::string("yes"));
    }

    //! build an mdp options tree that sets the options for the qmmm module
    void makeQMMMModuleWithSetOptions()
    {
        KeyValueTreeObject mdpOptionsTree = mdpValueBuilder_.build();

        QMMMModule_ = QMMMModuleInfo::create();

        // set up options
        Options QMMMModuleOptions;
        QMMMModule_->mdpOptionProvider()->initMdpOptions(&QMMMModuleOptions);

        // Add rules to transform mdp inputs to QMMMModule data
        KeyValueTreeTransformer transform;
        transform.rules()->addRule().keyMatchType("/", StringCompareType::CaseAndDashInsensitive);
        QMMMModule_->mdpOptionProvider()->initMdpTransform(transform.rules());

        // Execute the transform on the mdpValues
        auto transformedMdpValues = transform.transform(mdpOptionsTree, nullptr);
        assignOptionsFromKeyValueTree(&QMMMModuleOptions, transformedMdpValues.object(), nullptr);
    }

    void initializeForceProviders() { QMMMModule_->initForceProviders(&QMMMForces_); }

protected:
    KeyValueTreeBuilder        mdpValueBuilder_;
    ForceProviders             QMMMForces_;
    std::unique_ptr<IMDModule> QMMMModule_;
};

TEST_F(QMMMTest, ForceProviderLackingInputThrows)
{
    // Prepare MDP inputs
    addMdpOptionQMMMActive();

    // Initialize with default options
    makeQMMMModuleWithSetOptions();

    // Build the force provider, once all input data is gathered
    EXPECT_ANY_THROW(initializeForceProviders());
}

} // namespace

} // namespace gmx
