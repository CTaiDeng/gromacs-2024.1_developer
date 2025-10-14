/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2015- The GROMACS Authors
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
 * Declares functions for reference data XML persistence.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_testutils
 */
#ifndef GMX_TESTUTILS_REFDATA_XML_H
#define GMX_TESTUTILS_REFDATA_XML_H

#include <string>

#include "refdata_impl.h"

namespace gmx
{
namespace test
{

class ReferenceDataEntry;

//! \cond internal
/*! \internal
 * \brief
 * Loads reference data from an XML file.
 *
 * \param[in] path  Path to the file from which the data is loaded.
 * \returns   Root entry for the reference data parsed from the file.
 * \throws    TestException if there is a problem reading the file.
 *
 * \ingroup module_testutils
 */
ReferenceDataEntry::EntryPointer readReferenceDataFile(const std::string& path);
/*! \internal
 * \brief
 * Saves reference data to an XML file.
 *
 * \param[in] path  Path to the file to which the data is saved.
 * \param[in] root  Root entry for the reference data to write.
 * \throws    TestException if there is a problem writing the file.
 *
 * \ingroup module_testutils
 */
void writeReferenceDataFile(const std::string& path, const ReferenceDataEntry& root);
//! \endcond

} // namespace test
} // namespace gmx

#endif
