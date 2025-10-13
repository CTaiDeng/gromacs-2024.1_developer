/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2023- The GROMACS Authors
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
 * Stub implementation of ColvarsMDModule
 * Compiled in case if Colvars is not activated with CMake.
 *
 * \author Hubert Santuz <hubert.santuz@gmail.com>
 * \ingroup module_applied_forces
 */
#include "gmxpre.h"

#include <memory>
#include <string>

#include "gromacs/domdec/localatomsetmanager.h"
#include "gromacs/fileio/checkpoint.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/imdmodule.h"
#include "gromacs/mdtypes/imdpoptionprovider.h"
#include "gromacs/utility/keyvaluetreebuilder.h"

#include "colvarsMDModule.h"


namespace gmx
{

namespace
{

/*! \internal
 * \brief Colvars Options
 *
 * Stub Implementation in case Colvars library is not compiled
 */
class ColvarsOptions final : public IMdpOptionProvider
{

private:
    const std::string c_colvarsModuleName = "colvars";
    const std::string c_activeTag_        = "active";

public:
    void initMdpTransform(IKeyValueTreeTransformRules* /*rules*/) override {}

    /*! \brief Create Colvars mdp output even if Colvars is not compiled.
     * This is to be consistent with the case where Colvars is compiled but not activated
     */
    void buildMdpOutput(KeyValueTreeObjectBuilder* builder) const override
    {
        // new empty line before writing colvars mdp values
        builder->addValue<std::string>("comment-" + c_colvarsModuleName + "empty-line", "");

        builder->addValue<std::string>("comment-" + c_colvarsModuleName + "-module",
                                       "; Colvars bias");
        builder->addValue<bool>(c_colvarsModuleName + "-" + c_activeTag_, false);
    }

    void initMdpOptions(IOptionsContainerWithSections* /*options*/) override {}
};


/*! \internal
 * \brief Colvars module
 *
 * Stub Implementation in case Colvars library is not compiled
 */
class ColvarsMDModule final : public IMDModule
{
public:
    //! \brief Construct the colvars module.
    explicit ColvarsMDModule() = default;


    void subscribeToPreProcessingNotifications(MDModulesNotifiers* /*notifier*/) override {}

    void subscribeToSimulationSetupNotifications(MDModulesNotifiers* /*notifier*/) override {}

    IMdpOptionProvider* mdpOptionProvider() override { return &colvarsOptionsStub_; }

    IMDOutputProvider* outputProvider() override { return nullptr; }

    void initForceProviders(ForceProviders* /*forceProviders*/) override {}

private:
    ColvarsOptions colvarsOptionsStub_;
};


} // namespace

std::unique_ptr<IMDModule> ColvarsModuleInfo::create()
{
    return std::make_unique<ColvarsMDModule>();
}

const std::string ColvarsModuleInfo::name_ = "colvars";

} // namespace gmx
