/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2012- The GROMACS Authors
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
 * Implements functions in testoptions.h.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_testutils
 */
#include "gmxpre.h"

#include "testutils/testoptions.h"

#include <list>
#include <memory>
#include <mutex>

#include "gromacs/utility/classhelpers.h"

namespace gmx
{
namespace test
{

namespace
{

/*! \brief
 * Singleton registry for test options added with #GMX_TEST_OPTIONS.
 *
 * \ingroup module_testutils
 */
class TestOptionsRegistry
{
public:
    //! Returns the singleton instance of this class.
    static TestOptionsRegistry& getInstance()
    {
        static TestOptionsRegistry singleton;
        return singleton;
    }

    //! Adds a provider into the registry.
    void add(const char* /*name*/, TestOptionsProvider* provider)
    {
        std::lock_guard<std::mutex> lock(listMutex_);
        providerList_.push_back(provider);
    }

    //! Initializes the options from all the provides.
    void initOptions(IOptionsContainer* options);

private:
    TestOptionsRegistry() {}

    typedef std::list<TestOptionsProvider*> ProviderList;

    std::mutex   listMutex_;
    ProviderList providerList_;

    GMX_DISALLOW_COPY_AND_ASSIGN(TestOptionsRegistry);
};

void TestOptionsRegistry::initOptions(IOptionsContainer* options)
{
    // TODO: Have some deterministic order for the options; now it depends on
    // the order in which the global initializers are run.
    std::lock_guard<std::mutex>  lock(listMutex_);
    ProviderList::const_iterator i;
    for (i = providerList_.begin(); i != providerList_.end(); ++i)
    {
        (*i)->initOptions(options);
    }
}

} // namespace

void registerTestOptions(const char* name, TestOptionsProvider* provider)
{
    TestOptionsRegistry::getInstance().add(name, provider);
}

void initTestOptions(IOptionsContainer* options)
{
    TestOptionsRegistry::getInstance().initOptions(options);
}

} // namespace test
} // namespace gmx
