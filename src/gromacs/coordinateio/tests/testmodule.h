/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2019- The GROMACS Authors
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

/*!\file
 * \internal
 * \brief
 * Dummy module used for tests and as an implementation example.
 *
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 * \libinternal
 * \ingroup module_coordinateio
 */

#ifndef GMX_COORDINATEIO_TESTMODULE_H
#define GMX_COORDINATEIO_TESTMODULE_H

#include "gromacs/coordinateio/coordinatefileenums.h"
#include "gromacs/coordinateio/ioutputadapter.h"

namespace gmx
{

namespace test
{

class DummyOutputModule : public IOutputAdapter
{
public:
    explicit DummyOutputModule(CoordinateFileFlags requirementsFlag) :
        moduleRequirements_(requirementsFlag)
    {
    }

    DummyOutputModule(DummyOutputModule&& old) noexcept = default;

    ~DummyOutputModule() override {}

    void processFrame(int /*framenumber*/, t_trxframe* /*input*/) override {}

    void checkAbilityDependencies(unsigned long abilities) const override;

private:
    //! Local requirements
    CoordinateFileFlags moduleRequirements_;
};

} // namespace test

} // namespace gmx

#endif
