/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2020- The GROMACS Authors
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
 * \brief Defines the checkpoint data structure for the modular simulator
 *
 * \author Pascal Merz <pascal.merz@me.com>
 * \ingroup module_mdtypes
 */

#include "gmxpre.h"

#include "checkpointdata.h"

#include "gromacs/utility/iserializer.h"
#include "gromacs/utility/keyvaluetreeserializer.h"
#include "gromacs/utility/textwriter.h"

namespace gmx
{

void ReadCheckpointDataHolder::deserialize(ISerializer* serializer)
{
    GMX_RELEASE_ASSERT(serializer->reading(),
                       "Tried to deserialize using a serializing ISerializer object.");

    checkpointTree_ = deserializeKeyValueTree(serializer);
}

void WriteCheckpointDataHolder::serialize(ISerializer* serializer)
{
    GMX_RELEASE_ASSERT(!serializer->reading(),
                       "Tried to serialize using a deserializing ISerializer object.");

    serializeKeyValueTree(outputTreeBuilder_.build(), serializer);

    // Tree builder should not be used after build() (see docstring)
    // Make new builder to leave object in valid state
    outputTreeBuilder_ = KeyValueTreeBuilder();
}

bool ReadCheckpointDataHolder::keyExists(const std::string& key) const
{
    return checkpointTree_.keyExists(key);
}

std::vector<std::string> ReadCheckpointDataHolder::keys() const
{
    std::vector<std::string> keys;
    for (const auto& property : checkpointTree_.properties())
    {
        keys.emplace_back(property.key());
    }
    return keys;
}

ReadCheckpointData ReadCheckpointDataHolder::checkpointData(const std::string& key) const
{
    return ReadCheckpointData(checkpointTree_[key].asObject());
}

void ReadCheckpointDataHolder::dump(FILE* out) const
{
    if (out != nullptr)
    {
        TextWriter textWriter(out);
        dumpKeyValueTree(&textWriter, checkpointTree_);
    }
}

WriteCheckpointData WriteCheckpointDataHolder::checkpointData(const std::string& key)
{
    hasCheckpointDataBeenRequested_ = true;
    return WriteCheckpointData(outputTreeBuilder_.rootObject().addObject(key));
}

bool WriteCheckpointDataHolder::empty() const
{
    return !hasCheckpointDataBeenRequested_;
}

} // namespace gmx
