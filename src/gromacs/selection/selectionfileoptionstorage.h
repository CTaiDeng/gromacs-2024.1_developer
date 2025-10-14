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
 * Declares gmx::SelectionFileOptionStorage.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_selection
 */
#ifndef GMX_SELECTION_SELECTIONFILEOPTIONSTORAGE_H
#define GMX_SELECTION_SELECTIONFILEOPTIONSTORAGE_H

#include "gromacs/options/abstractoptionstorage.h"
#include "gromacs/selection/selectionfileoption.h"

namespace gmx
{

class SelectionFileOption;
class SelectionOptionManager;

/*! \internal \brief
 * Implementation for a special option for reading selections from files.
 *
 * \ingroup module_selection
 */
class SelectionFileOptionStorage : public AbstractOptionStorage
{
public:
    /*! \brief
     * Initializes the storage from option settings.
     *
     * \param[in] settings   Storage settings.
     * \param     manager    Manager for this object.
     */
    SelectionFileOptionStorage(const SelectionFileOption& settings, SelectionOptionManager* manager);

    OptionInfo&              optionInfo() override { return info_; }
    std::string              typeString() const override { return "file"; }
    int                      valueCount() const override { return 0; }
    std::vector<Any>         defaultValues() const override { return {}; }
    std::vector<std::string> defaultValuesAsStrings() const override { return {}; }
    std::vector<Any>         normalizeValues(const std::vector<Any>& values) const override
    {
        return values;
    }

private:
    void clearSet() override;
    void convertValue(const Any& value) override;
    void processSet() override;
    void processAll() override {}

    SelectionFileOptionInfo info_;
    SelectionOptionManager& manager_;
    bool                    bValueParsed_;
};

} // namespace gmx

#endif
