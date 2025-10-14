/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2019- The GROMACS Authors
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
 * Tests for mrc file data structure.
 *
 * \author Christian Blau <cblau@gwdg.de>
 * \ingroup module_fileio
 */
#include "gmxpre.h"

#include "gromacs/fileio/mrcserializer.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "gromacs/fileio/mrcdensitymapheader.h"
#include "gromacs/utility/inmemoryserializer.h"

#include "testutils/testasserts.h"

namespace gmx
{

namespace
{

TEST(MrcSerializer, DefaultHeaderValuesAreSet)
{
    const MrcDensityMapHeader inputHeader = {};

    EXPECT_EQ('M', inputHeader.formatIdentifier_[0]);
    EXPECT_EQ('A', inputHeader.formatIdentifier_[1]);
    EXPECT_EQ('P', inputHeader.formatIdentifier_[2]);
    EXPECT_EQ(' ', inputHeader.formatIdentifier_[3]);
}

TEST(MrcSerializer, DefaultHeaderHasRightSerialSize)
{

    InMemorySerializer        serializer;
    const MrcDensityMapHeader inputHeader = {};

    serializeMrcDensityMapHeader(&serializer, inputHeader);
    const auto serializedHeader = serializer.finishAndGetBuffer();

    constexpr size_t c_defaultMrcHeaderSize = 1024;
    EXPECT_EQ(c_defaultMrcHeaderSize, serializedHeader.size());
}

TEST(MrcSerializer, DefaultHeaderIdenticalAfterRoundTrip)
{
    InMemorySerializer        serializer;
    const MrcDensityMapHeader inputHeader = {};

    serializeMrcDensityMapHeader(&serializer, inputHeader);
    const auto serializedHeader = serializer.finishAndGetBuffer();

    InMemoryDeserializer deserializer(serializedHeader, false);
    const auto           deserializedHeader = deserializeMrcDensityMapHeader(&deserializer);

    // comparing serialized results saves MrcDensityHeaders comparison implementation
    serializeMrcDensityMapHeader(&serializer, deserializedHeader);
    const auto roundTripResult = serializer.finishAndGetBuffer();
    EXPECT_THAT(serializedHeader, testing::Pointwise(testing::Eq(), roundTripResult));
}

} // namespace

} // namespace gmx
