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

/*! \internal \file
 * \brief
 * Implements classes and functions from helptopic.h.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_onlinehelp
 */
#include "gmxpre.h"

#include "helptopic.h"

#include <map>
#include <utility>

#include "gromacs/onlinehelp/helpformat.h"
#include "gromacs/onlinehelp/helpwritercontext.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/stringutil.h"
#include "gromacs/utility/textwriter.h"

namespace gmx
{

/********************************************************************
 * AbstractSimpleHelpTopic
 */

bool AbstractSimpleHelpTopic::hasSubTopics() const
{
    return false;
}

const IHelpTopic* AbstractSimpleHelpTopic::findSubTopic(const char* /* name */) const
{
    return nullptr;
}

void AbstractSimpleHelpTopic::writeHelp(const HelpWriterContext& context) const
{
    context.writeTextBlock(helpText());
}

/********************************************************************
 * AbstractCompositeHelpTopic::Impl
 */

/*! \internal \brief
 * Private implementation class for AbstractCompositeHelpTopic.
 *
 * \ingroup module_onlinehelp
 */
class AbstractCompositeHelpTopic::Impl
{
public:
    //! Container for subtopics.
    typedef std::vector<HelpTopicPointer> SubTopicList;
    //! Container for mapping subtopic names to help topic objects.
    typedef std::map<std::string, const IHelpTopic*> SubTopicMap;

    /*! \brief
     * Subtopics in the order they were added.
     *
     * Owns the contained subtopics.
     */
    SubTopicList subTopics_;
    /*! \brief
     * Maps subtopic names to help topic objects.
     *
     * Points to objects in the \a subTopics_ map.
     */
    SubTopicMap subTopicMap_;
};

/********************************************************************
 * AbstractCompositeHelpTopic
 */

AbstractCompositeHelpTopic::AbstractCompositeHelpTopic() : impl_(new Impl) {}

AbstractCompositeHelpTopic::~AbstractCompositeHelpTopic() {}

bool AbstractCompositeHelpTopic::hasSubTopics() const
{
    return !impl_->subTopics_.empty();
}

const IHelpTopic* AbstractCompositeHelpTopic::findSubTopic(const char* name) const
{
    Impl::SubTopicMap::const_iterator topic = impl_->subTopicMap_.find(name);
    if (topic == impl_->subTopicMap_.end())
    {
        return nullptr;
    }
    return topic->second;
}

void AbstractCompositeHelpTopic::writeHelp(const HelpWriterContext& context) const
{
    context.writeTextBlock(helpText());
    writeSubTopicList(context, "\nAvailable subtopics:");
}

bool AbstractCompositeHelpTopic::writeSubTopicList(const HelpWriterContext& context,
                                                   const std::string&       title) const
{
    if (context.outputFormat() != eHelpOutputFormat_Console)
    {
        Impl::SubTopicList::const_iterator topic;
        for (topic = impl_->subTopics_.begin(); topic != impl_->subTopics_.end(); ++topic)
        {
            const char* const topic_title = (*topic)->title();
            if (!isNullOrEmpty(topic_title))
            {
                context.paragraphBreak();
                HelpWriterContext subContext(context);
                subContext.enterSubSection(topic_title);
                (*topic)->writeHelp(subContext);
            }
        }
        return true;
    }
    int                               maxNameLength = 0;
    Impl::SubTopicMap::const_iterator topic;
    for (topic = impl_->subTopicMap_.begin(); topic != impl_->subTopicMap_.end(); ++topic)
    {
        const char* const topic_title = topic->second->title();
        if (!isNullOrEmpty(topic_title))
        {
            int nameLength = static_cast<int>(topic->first.length());
            if (nameLength > maxNameLength)
            {
                maxNameLength = nameLength;
            }
        }
    }
    if (maxNameLength == 0)
    {
        return false;
    }
    TextWriter&        file = context.outputFile();
    TextTableFormatter formatter;
    formatter.addColumn(nullptr, maxNameLength + 1, false);
    formatter.addColumn(nullptr, 72 - maxNameLength, true);
    formatter.setFirstColumnIndent(4);
    file.writeLine(title);
    for (topic = impl_->subTopicMap_.begin(); topic != impl_->subTopicMap_.end(); ++topic)
    {
        const char* const topicName  = topic->first.c_str();
        const char* const topicTitle = topic->second->title();
        if (!isNullOrEmpty(topicTitle))
        {
            formatter.clear();
            formatter.addColumnLine(0, topicName);
            formatter.addColumnLine(1, topicTitle);
            file.writeString(formatter.formatRow());
        }
    }
    return true;
}

void AbstractCompositeHelpTopic::addSubTopic(HelpTopicPointer topic)
{
    GMX_ASSERT(impl_->subTopicMap_.find(topic->name()) == impl_->subTopicMap_.end(),
               "Attempted to register a duplicate help topic name");
    const IHelpTopic* topicPtr = topic.get();
    impl_->subTopics_.reserve(impl_->subTopics_.size() + 1);
    impl_->subTopicMap_.insert(std::make_pair(std::string(topicPtr->name()), topicPtr));
    impl_->subTopics_.push_back(std::move(topic));
}

} // namespace gmx
