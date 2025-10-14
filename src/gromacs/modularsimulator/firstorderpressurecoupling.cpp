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
 * \brief Defines the element performing first-order pressure coupling for the modular simulator
 *
 * \author Pascal Merz <pascal.merz@me.com>
 * \ingroup module_modularsimulator
 */

#include "gmxpre.h"

#include "firstorderpressurecoupling.h"

#include "gromacs/domdec/domdec_network.h"
#include "gromacs/mdlib/coupling.h"
#include "gromacs/mdlib/mdatoms.h"
#include "gromacs/mdlib/stat.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/group.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/mdtypes/mdatom.h"
#include "gromacs/pbcutil/boxutilities.h"

#include "energydata.h"
#include "simulatoralgorithm.h"
#include "statepropagatordata.h"

namespace gmx
{

template<PressureCoupling pressureCouplingType>
void FirstOrderPressureCoupling::calculateScalingMatrix(Step step)
{
    const auto& ekindata         = *energyData_->ekindata();
    const auto* pressure         = energyData_->pressure(step);
    const auto* forceVirial      = energyData_->forceVirial(step);
    const auto* constraintVirial = energyData_->constraintVirial(step);
    const auto* box              = statePropagatorData_->constBox();

    const real ensembleTemperature =
            (haveEnsembleTemperature(*inputrec_) ? ekindata.currentEnsembleTemperature() : 0.0_real);

    previousStepConservedEnergyContribution_ = conservedEnergyContribution_;
    pressureCouplingCalculateScalingMatrix<pressureCouplingType>(fplog_,
                                                                 step,
                                                                 inputrec_->pressureCouplingOptions,
                                                                 inputrec_->ld_seed,
                                                                 ensembleTemperature,
                                                                 couplingTimeStep_,
                                                                 pressure,
                                                                 box,
                                                                 forceVirial,
                                                                 constraintVirial,
                                                                 &boxScalingMatrix_,
                                                                 &conservedEnergyContribution_);
    conservedEnergyContributionStep_ = step;
}

template<PressureCoupling pressureCouplingType>
void FirstOrderPressureCoupling::scaleBoxAndCoordinates()
{
    auto*          box       = statePropagatorData_->box();
    auto           positions = statePropagatorData_->positionsView().unpaddedArrayRef();
    ArrayRef<RVec> velocities;
    if (pressureCouplingType == PressureCoupling::CRescale)
    {
        velocities = statePropagatorData_->velocitiesView().unpaddedArrayRef();
    }
    // Freeze groups are not implemented
    ArrayRef<const unsigned short> cFreeze;
    // Coordinates are always scaled except for GPU update (not implemented currently)
    const bool scaleCoordinates = true;
    // Atom range
    const int startAtom = 0;
    const int numAtoms  = mdAtoms_->mdatoms()->homenr;

    pressureCouplingScaleBoxAndCoordinates<pressureCouplingType>(inputrec_->pressureCouplingOptions,
                                                                 inputrec_->deform,
                                                                 inputrec_->opts.nFreeze,
                                                                 boxScalingMatrix_,
                                                                 box,
                                                                 boxRel_,
                                                                 startAtom,
                                                                 numAtoms,
                                                                 positions,
                                                                 velocities,
                                                                 cFreeze,
                                                                 nrnb_,
                                                                 scaleCoordinates);
}

void FirstOrderPressureCoupling::scheduleTask(Step step, Time /*unused*/, const RegisterRunFunction& registerRunFunction)
{
    if (do_per_step(step + couplingFrequency_ + couplingOffset_, couplingFrequency_))
    {
        if (pressureCouplingType_ == PressureCoupling::Berendsen)
        {
            registerRunFunction([this, step]() {
                calculateScalingMatrix<PressureCoupling::Berendsen>(step);
                scaleBoxAndCoordinates<PressureCoupling::Berendsen>();
            });
        }
        else if (pressureCouplingType_ == PressureCoupling::CRescale)
        {
            registerRunFunction([this, step]() {
                calculateScalingMatrix<PressureCoupling::CRescale>(step);
                scaleBoxAndCoordinates<PressureCoupling::CRescale>();
            });
        }
    }
}

void FirstOrderPressureCoupling::elementSetup()
{
    if (shouldPreserveBoxShape(inputrec_->pressureCouplingOptions, inputrec_->deform))
    {
        auto*     box = statePropagatorData_->box();
        const int ndim =
                inputrec_->pressureCouplingOptions.epct == PressureCouplingType::SemiIsotropic ? 2 : 3;
        do_box_rel(ndim, inputrec_->deform, boxRel_, box, true);
    }
}

real FirstOrderPressureCoupling::conservedEnergyContribution(Step step)
{
    if (step == conservedEnergyContributionStep_
        && reportPreviousStepConservedEnergy_ == ReportPreviousStepConservedEnergy::Yes)
    {
        return previousStepConservedEnergyContribution_;
    }
    return conservedEnergyContribution_;
}

namespace
{
/*!
 * \brief Enum describing the contents FirstOrderPressureCoupling writes to modular checkpoint
 *
 * When changing the checkpoint content, add a new element just above Count, and adjust the
 * checkpoint functionality.
 */
enum class CheckpointVersion
{
    Base, //!< First version of modular checkpointing
    Count //!< Number of entries. Add new versions right above this!
};
constexpr auto c_currentVersion = CheckpointVersion(int(CheckpointVersion::Count) - 1);
} // namespace

template<CheckpointDataOperation operation>
void FirstOrderPressureCoupling::doCheckpointData(CheckpointData<operation>* checkpointData)
{
    checkpointVersion(checkpointData, "FirstOrderPressureCoupling version", c_currentVersion);

    checkpointData->scalar("conserved energy contribution", &conservedEnergyContribution_);
    checkpointData->scalar("conserved energy step", &conservedEnergyContributionStep_);
    checkpointData->tensor("relative box vector", boxRel_);
}

void FirstOrderPressureCoupling::saveCheckpointState(std::optional<WriteCheckpointData> checkpointData,
                                                     const t_commrec*                   cr)
{
    if (MAIN(cr))
    {
        doCheckpointData<CheckpointDataOperation::Write>(&checkpointData.value());
    }
}

void FirstOrderPressureCoupling::restoreCheckpointState(std::optional<ReadCheckpointData> checkpointData,
                                                        const t_commrec*                  cr)
{
    if (MAIN(cr))
    {
        doCheckpointData<CheckpointDataOperation::Read>(&checkpointData.value());
    }
    if (haveDDAtomOrdering(*cr))
    {
        dd_bcast(cr->dd, sizeof(conservedEnergyContribution_), &conservedEnergyContribution_);
        dd_bcast(cr->dd, sizeof(conservedEnergyContributionStep_), &conservedEnergyContributionStep_);
        dd_bcast(cr->dd, sizeof(boxRel_), boxRel_);
    }
}

const std::string& FirstOrderPressureCoupling::clientID()
{
    return identifier_;
}

FirstOrderPressureCoupling::FirstOrderPressureCoupling(int                  couplingFrequency,
                                                       int                  couplingOffset,
                                                       real                 couplingTimeStep,
                                                       StatePropagatorData* statePropagatorData,
                                                       EnergyData*          energyData,
                                                       FILE*                fplog,
                                                       const t_inputrec*    inputrec,
                                                       const MDAtoms*       mdAtoms,
                                                       t_nrnb*              nrnb,
                                                       ReportPreviousStepConservedEnergy reportPreviousStepConservedEnergy) :
    pressureCouplingType_(inputrec->pressureCouplingOptions.epc),
    couplingTimeStep_(couplingTimeStep),
    couplingFrequency_(couplingFrequency),
    couplingOffset_(couplingOffset),
    boxScalingMatrix_{ { 0 } },
    boxRel_{ { 0 } },
    conservedEnergyContribution_(0),
    previousStepConservedEnergyContribution_(0),
    conservedEnergyContributionStep_(-1),
    reportPreviousStepConservedEnergy_(reportPreviousStepConservedEnergy),
    statePropagatorData_(statePropagatorData),
    energyData_(energyData),
    fplog_(fplog),
    inputrec_(inputrec),
    mdAtoms_(mdAtoms),
    nrnb_(nrnb),
    identifier_("FirstOrderPressureCoupling-" + std::string(enumValueToString(pressureCouplingType_)))
{
    energyData->addConservedEnergyContribution(
            [this](Step step, Time /*unused*/) { return conservedEnergyContribution(step); });
}

ISimulatorElement* FirstOrderPressureCoupling::getElementPointerImpl(
        LegacySimulatorData*                    legacySimulatorData,
        ModularSimulatorAlgorithmBuilderHelper* builderHelper,
        StatePropagatorData*                    statePropagatorData,
        EnergyData*                             energyData,
        FreeEnergyPerturbationData gmx_unused* freeEnergyPerturbationData,
        GlobalCommunicationHelper gmx_unused* globalCommunicationHelper,
        ObservablesReducer* /*observablesReducer*/,
        int                               offset,
        ReportPreviousStepConservedEnergy reportPreviousStepConservedEnergy)
{
    return builderHelper->storeElement(std::make_unique<FirstOrderPressureCoupling>(
            legacySimulatorData->inputRec_->pressureCouplingOptions.nstpcouple,
            offset,
            legacySimulatorData->inputRec_->delta_t
                    * legacySimulatorData->inputRec_->pressureCouplingOptions.nstpcouple,
            statePropagatorData,
            energyData,
            legacySimulatorData->fpLog_,
            legacySimulatorData->inputRec_,
            legacySimulatorData->mdAtoms_,
            legacySimulatorData->nrnb_,
            reportPreviousStepConservedEnergy));
}

} // namespace gmx
