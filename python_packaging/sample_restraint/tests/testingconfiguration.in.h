/*
 * Copyright (C) 2025 GaoZheng
 * SPDX-License-Identifier: GPL-3.0-only
 * This file is part of this project.
 * Licensed under the GNU General Public License version 3.
 * See https://www.gnu.org/licenses/gpl-3.0.html for details.
 */
//
// Created by Eric Irrgang on 11/9/17.
//

#ifndef HARMONICRESTRAINT_TESTINGCONFIGURATION_IN_H
#define HARMONICRESTRAINT_TESTINGCONFIGURATION_IN_H


#include <string>

namespace plugin
{

namespace testing
{

// Todo: Need to set up a test fixture...
static const std::string sample_tprfilename = "${CMAKE_CURRENT_BINARY_DIR}/topol.tpr";

} // namespace testing

} // end namespace plugin


#endif // HARMONICRESTRAINT_TESTINGCONFIGURATION_IN_H
