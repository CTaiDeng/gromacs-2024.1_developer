/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2016- The GROMACS Authors
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

/*! \libinternal \file
 * \brief
 * Declares gmx::ISerializer implementation for in-memory serialization.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \inlibraryapi
 * \ingroup module_utility
 */
#ifndef GMX_UTILITY_INMEMORYSERIALIZER_H
#define GMX_UTILITY_INMEMORYSERIALIZER_H

#include <cstddef>

#include <memory>
#include <vector>

#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/iserializer.h"

namespace gmx
{

//! Specify endian swapping behavoir.
//
// The host-dependent choices avoid the calling file having to
// depend on config.h.
//
enum class EndianSwapBehavior : int
{
    DoNotSwap,                //!< Don't touch anything
    Swap,                     //!< User-enforced swapping
    SwapIfHostIsBigEndian,    //!< Only swap if machine we execute on is big-endian
    SwapIfHostIsLittleEndian, //!< Only swap if machine we execute on is little-endian
    Count                     //!< Number of possible behaviors
};

class InMemorySerializer : public ISerializer
{
public:
    explicit InMemorySerializer(EndianSwapBehavior endianSwapBehavior = EndianSwapBehavior::DoNotSwap);
    ~InMemorySerializer() override;

    std::vector<char> finishAndGetBuffer();

    // From ISerializer
    bool reading() const override { return false; }
    void doBool(bool* value) override;
    void doUChar(unsigned char* value) override;
    void doChar(char* value) override;
    void doUShort(unsigned short* value) override;
    void doInt(int* value) override;
    void doInt32(int32_t* value) override;
    void doInt64(int64_t* value) override;
    void doFloat(float* value) override;
    void doDouble(double* value) override;
    void doReal(real* value) override;
    void doIvec(ivec* value) override;
    void doRvec(rvec* value) override;
    void doString(std::string* value) override;
    void doOpaque(char* data, std::size_t size) override;

private:
    class Impl;

    std::unique_ptr<Impl> impl_;
};

class InMemoryDeserializer : public ISerializer
{
public:
    InMemoryDeserializer(ArrayRef<const char> buffer,
                         bool                 sourceIsDouble,
                         EndianSwapBehavior   endianSwapBehavior = EndianSwapBehavior::DoNotSwap);
    ~InMemoryDeserializer() override;

    //! Get if the source data was written in double precsion
    bool sourceIsDouble() const;

    // From ISerializer
    bool reading() const override { return true; }
    void doBool(bool* value) override;
    void doUChar(unsigned char* value) override;
    void doChar(char* value) override;
    void doUShort(unsigned short* value) override;
    void doInt(int* value) override;
    void doInt32(int32_t* value) override;
    void doInt64(int64_t* value) override;
    void doFloat(float* value) override;
    void doDouble(double* value) override;
    void doReal(real* value) override;
    void doIvec(ivec* value) override;
    void doRvec(rvec* value) override;
    void doString(std::string* value) override;
    void doOpaque(char* data, std::size_t size) override;

private:
    class Impl;

    std::unique_ptr<Impl> impl_;
};

} // namespace gmx

#endif
