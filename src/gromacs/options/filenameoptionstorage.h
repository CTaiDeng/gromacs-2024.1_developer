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
 * Declares gmx::FileNameOptionStorage.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_options
 */
#ifndef GMX_OPTIONS_FILENAMEOPTIONSTORAGE_H
#define GMX_OPTIONS_FILENAMEOPTIONSTORAGE_H

#include <string>
#include <vector>

#include "gromacs/options/filenameoption.h"
#include "gromacs/options/optionfiletype.h"
#include "gromacs/options/optionstoragetemplate.h"

namespace gmx
{

class FileNameOption;
class FileNameOptionManager;

/*! \internal \brief
 * Converts, validates, and stores file names.
 */
class FileNameOptionStorage : public OptionStorageTemplateSimple<std::string>
{
public:
    /*! \brief
     * Initializes the storage from option settings.
     *
     * \param[in] settings   Storage settings.
     * \param     manager    Manager for this object (can be NULL).
     */
    FileNameOptionStorage(const FileNameOption& settings, FileNameOptionManager* manager);

    OptionInfo& optionInfo() override { return info_; }
    std::string typeString() const override;
    std::string formatExtraDescription() const override;
    std::string formatSingleValue(const std::string& value) const override;

    //! \copydoc FileNameOptionInfo::isInputFile()
    bool isInputFile() const { return bRead_ && !bWrite_; }
    //! \copydoc FileNameOptionInfo::isOutputFile()
    bool isOutputFile() const { return !bRead_ && bWrite_; }
    //! \copydoc FileNameOptionInfo::isInputOutputFile()
    bool isInputOutputFile() const { return bRead_ && bWrite_; }
    //! \copydoc FileNameOptionInfo::isLibraryFile()
    bool isLibraryFile() const { return bLibrary_; }
    //! \copydoc FileNameOptionInfo::allowMissing()
    bool allowMissing() const { return bAllowMissing_; }

    //! \copydoc FileNameOptionInfo::isDirectoryOption()
    bool isDirectoryOption() const;
    //! \copydoc FileNameOptionInfo::isTrajectoryOption()
    bool isTrajectoryOption() const;
    //! \copydoc FileNameOptionInfo::defaultExtension()
    const char* defaultExtension() const;
    //! \copydoc FileNameOptionInfo::extensions()
    std::vector<const char*> extensions() const;
    //! \copydoc FileNameOptionInfo::isValidType()
    bool isValidType(int fileType) const;
    //! \copydoc FileNameOptionInfo::fileTypes()
    ArrayRef<const int> fileTypes() const;

private:
    void        initConverter(ConverterType* converter) override;
    std::string processValue(const std::string& value) const override;
    void        processAll() override;

    FileNameOptionInfo     info_;
    FileNameOptionManager* manager_;
    int                    fileType_;
    const char*            defaultExtension_;
    bool                   bRead_;
    bool                   bWrite_;
    bool                   bLibrary_;
    bool                   bAllowMissing_;
};

} // namespace gmx

#endif
