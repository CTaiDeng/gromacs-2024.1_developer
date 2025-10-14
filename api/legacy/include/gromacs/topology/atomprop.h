/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 1991- The GROMACS Authors
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

#ifndef GMX_TOPOLOGY_ATOMPROP_H
#define GMX_TOPOLOGY_ATOMPROP_H

#include <memory>
#include <string>

#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/real.h"

enum
{
    epropMass,
    epropVDW,
    epropDGsol,
    epropElectroneg,
    epropElement,
    epropNR
};

struct AtomProperty;
/*! \brief
 * Holds all the atom property information loaded.
 */
class AtomProperties
{
public:
    //! Default constructor.
    AtomProperties();
    //! Default destructor
    ~AtomProperties();

    /*! \brief
     * Get element string from atom number.
     *
     * \param[in] atomNumber Atomnumber to check.
     * \returns Name of the element.
     *
     * \todo This should be made const once the lazy
     * implementation is done properly for the class.
     */
    std::string elementFromAtomNumber(int atomNumber);
    /*! \brief
     * Get atom number from element string.
     *
     * \param[in] element Name of element.
     * \returns AtomNumber that was being looked for.
     *
     * \todo This should be made const once the lazy
     * implementation is done properly for the class.
     */
    int atomNumberFromElement(const char* element);
    /*! \brief
     * Set atom property based on atomname.
     *
     * Extract a \p value from the database. Returns true
     * if this is successful, or false if not. Sets default value
     * in the later case. The first time this function is called
     * for this property the database will be initialized.
     *
     * \param[in] eprop Property to set.
     * \param[in] residueName Residue name for entry.
     * \param[in] atomName Atom name for entry.
     * \param[out] value New value to set or default.
     * \returns If the operation has been succesful.
     */
    bool setAtomProperty(int eprop, const std::string& residueName, const std::string& atomName, real* value);
    /*! \brief
     * Get handle to property.
     *
     * \param[in] eprop Which property we need a handle to.
     * \returns Pointer to property entry.
     */
    AtomProperty* prop(int eprop);

private:
    //! Implementation pointer.
    class Impl;

    std::unique_ptr<Impl> impl_;
};

#endif
