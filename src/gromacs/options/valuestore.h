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

/*! \file
 * \brief
 * Declares implementations for IOptionValueStore.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_options
 */
#ifndef GMX_OPTIONS_VALUESTORE_H
#define GMX_OPTIONS_VALUESTORE_H

#include <vector>

#include "gromacs/options/ivaluestore.h"
#include "gromacs/utility/arrayref.h"

namespace gmx
{

template<typename T>
class OptionValueStorePlain : public IOptionValueStore<T>
{
public:
    OptionValueStorePlain(T* store, int* storeCount, int initialCount) :
        count_(initialCount), store_(store), storeCount_(storeCount)
    {
    }

    int         valueCount() override { return count_; }
    ArrayRef<T> values() override { return arrayRefFromArray(store_, count_); }
    void        clear() override
    {
        count_ = 0;
        if (storeCount_ != nullptr)
        {
            *storeCount_ = count_;
        }
    }
    void reserve(size_t /*count*/) override {}
    void append(const T& value) override
    {
        store_[count_] = value;
        ++count_;
        if (storeCount_ != nullptr)
        {
            *storeCount_ = count_;
        }
    }

private:
    int  count_;
    T*   store_;
    int* storeCount_;
};

template<typename T>
class OptionValueStoreVector : public IOptionValueStore<T>
{
public:
    explicit OptionValueStoreVector(std::vector<T>* store) : store_(store) {}

    int         valueCount() override { return static_cast<int>(store_->size()); }
    ArrayRef<T> values() override { return *store_; }
    void        clear() override { store_->clear(); }
    void        reserve(size_t count) override { store_->reserve(store_->size() + count); }
    void        append(const T& value) override { store_->push_back(value); }

private:
    std::vector<T>* store_;
};

// Specialization that works around std::vector<bool> specialities.
template<>
class OptionValueStoreVector<bool> : public IOptionValueStore<bool>
{
public:
    explicit OptionValueStoreVector(std::vector<bool>* store) : store_(store) {}

    int            valueCount() override { return static_cast<int>(store_->size()); }
    ArrayRef<bool> values() override
    {
        return arrayRefFromArray(reinterpret_cast<bool*>(boolStore_.data()), boolStore_.size());
    }
    void clear() override
    {
        boolStore_.clear();
        store_->clear();
    }
    void reserve(size_t count) override
    {
        boolStore_.reserve(boolStore_.size() + count);
        store_->reserve(store_->size() + count);
    }
    void append(const bool& value) override
    {
        boolStore_.push_back({ value });
        store_->push_back(value);
    }

private:
    struct Bool
    {
        bool value;
    };

    std::vector<Bool>  boolStore_;
    std::vector<bool>* store_;
};

/*! \internal
 * \brief
 * Value storage that does not store anywhere.
 *
 * This is needed because even though the values are not stored anywhere, the
 * code still expects to access them later through valueCount() and values().
 *
 * \ingroup module_options
 */
template<typename T>
class OptionValueStoreNull : public IOptionValueStore<T>
{
public:
    OptionValueStoreNull() : store_(&vector_) {}

    int         valueCount() override { return store_.valueCount(); }
    ArrayRef<T> values() override { return store_.values(); }
    void        clear() override { store_.clear(); }
    void        reserve(size_t count) override { store_.reserve(count); }
    void        append(const T& value) override { store_.append(value); }

private:
    std::vector<T>            vector_;
    OptionValueStoreVector<T> store_;
};

} // namespace gmx

#endif
