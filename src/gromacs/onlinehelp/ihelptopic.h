/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2012- The GROMACS Authors
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
 * Declares gmx::IHelpTopic.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \inlibraryapi
 * \ingroup module_onlinehelp
 */
#ifndef GMX_ONLINEHELP_IHELPTOPIC_H
#define GMX_ONLINEHELP_IHELPTOPIC_H

#include <memory>

namespace gmx
{

class HelpWriterContext;

/*! \libinternal \brief
 * Provides a single online help topic.
 *
 * Implementations of these methods should not throw, except that writeHelp()
 * is allowed to throw on out-of-memory or I/O errors since those it cannot
 * avoid.
 *
 * Header helptopic.h contains classes that implement this interface and make
 * it simple to write concrete help topic classes.
 *
 * \inlibraryapi
 * \ingroup module_onlinehelp
 */
class IHelpTopic
{
public:
    virtual ~IHelpTopic() {}

    /*! \brief
     * Returns the name of the topic.
     *
     * This should be a single lowercase word, used to identify the topic.
     * It is not used for the root of the help topic tree.
     */
    virtual const char* name() const = 0;
    /*! \brief
     * Returns a title for the topic.
     *
     * May return NULL, in which case the topic is omitted from normal
     * subtopic lists and no title is printed by the methods provided in
     * helptopic.h.
     */
    virtual const char* title() const = 0;

    //! Returns whether the topic has any subtopics.
    virtual bool hasSubTopics() const = 0;
    /*! \brief
     * Finds a subtopic by name.
     *
     * \param[in] name  Name of subtopic to find.
     * \returns   Pointer to the found subtopic, or NULL if matching topic
     *      is not found.
     */
    virtual const IHelpTopic* findSubTopic(const char* name) const = 0;

    /*! \brief
     * Prints the help text for this topic.
     *
     * \param[in] context  Context object for writing the help.
     * \throws    std::bad_alloc if out of memory.
     * \throws    FileIOError on any I/O error.
     */
    virtual void writeHelp(const HelpWriterContext& context) const = 0;
};

//! Smart pointer type to manage a IHelpTopic object.
typedef std::unique_ptr<IHelpTopic> HelpTopicPointer;

} // namespace gmx

#endif
