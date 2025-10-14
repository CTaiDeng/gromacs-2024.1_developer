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

/*! \libinternal \defgroup module_onlinehelp Help Formatting for Online Help (onlinehelp)
 * \ingroup group_utilitymodules
 * \brief
 * Provides functionality for formatting help text for console and reStructuredText.
 *
 * This module provides helper functions and classes for formatting text to the
 * console and as reStructuredText through a single interface.  The main
 * components of the module are:
 *  - gmx::HelpWriterContext provides a single interface that can produce both
 *    output formats from the same input strings and API calls.  Whenever
 *    possible, the output format should be abstracted using this interface,
 *    but in some cases code still writes out raw reStructuredText.
 *  - rstparser.h provides the functionality to parse reStructuredText such that
 *    it can be rewrapped for console output.
 *  - helpformat.h provides some general text-processing classes, currently
 *    focused on producing aligned tables for console output.
 *  - ihelptopic.h, helptopic.h, and helpmanager.h provide classes for
 *    managing a hierarchy of help topics and printing out help from this
 *    hierarchy.
 *
 * The formatting syntax for strings accepted by this module is described in
 * \ref page_onlinehelp.  The module is currently exposed outside \Gromacs only
 * through this formatting syntax, not any API calls.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 */
/*! \internal \file
 * \brief
 * Dummy header for \ref module_onlinehelp documentation.
 */
