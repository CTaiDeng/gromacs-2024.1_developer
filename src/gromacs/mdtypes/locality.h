/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2019- The GROMACS Authors
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
 * \brief Defines atom and atom interaction locality enums
 *
 * \author Berk Hess <hess@kth.se>
 * \ingroup module_mdtypes
 */

#ifndef GMX_MDTYPES_LOCALITY_H
#define GMX_MDTYPES_LOCALITY_H

#include "gromacs/utility/enumerationhelpers.h"
#include "gromacs/utility/exceptions.h"

namespace gmx
{

/*! \brief Atom locality indicator: local, non-local, all.
 *
 * Used for calls to:
 * gridding, force calculation, x/f buffer operations
 */
enum class AtomLocality : int
{
    Local    = 0, //!< Local atoms
    NonLocal = 1, //!< Non-local atoms
    All      = 2, //!< Both local and non-local atoms
    Count    = 3  //!< The number of atom locality types
};

/*! \brief Get the human-friendly name for atom localities.
 *
 * \param[in] enumValue The enum value to get the name for.
 */
[[maybe_unused]] static const char* enumValueToString(AtomLocality enumValue)
{
    static constexpr gmx::EnumerationArray<AtomLocality, const char*> atomLocalityNames = {
        "Local", "Non-local", "All"
    };
    return atomLocalityNames[enumValue];
}
/*! \brief Interaction locality indicator: local, non-local, all.
 *
 * Used for calls to:
 * pair-search, force calculation, x/f buffer operations
 */
enum class InteractionLocality : int
{
    Local    = 0, //!< Interactions between local atoms only
    NonLocal = 1, //!< Interactions between non-local and (non-)local atoms
    Count    = 2  //!< The number of interaction locality types
};

/*! \brief Get the human-friendly name for interaction localities.
 *
 * \param[in] enumValue The enum value to get the name for.
 */
[[maybe_unused]] static const char* enumValueToString(InteractionLocality enumValue)
{
    static constexpr gmx::EnumerationArray<InteractionLocality, const char*> interactionLocalityNames = {
        "Local", "Non-local"
    };
    return interactionLocalityNames[enumValue];
}

/*! \brief Convert atom locality to interaction locality.
 *
 *  In the current implementation the this is straightforward conversion:
 *  local to local, non-local to non-local.
 *
 *  \param[in] atomLocality Atom locality specifier
 *  \returns                Interaction locality corresponding to the atom locality passed.
 */
static inline InteractionLocality atomToInteractionLocality(const AtomLocality atomLocality)
{

    /* determine interaction locality from atom locality */
    if (atomLocality == AtomLocality::Local)
    {
        return InteractionLocality::Local;
    }
    else if (atomLocality == AtomLocality::NonLocal)
    {
        return InteractionLocality::NonLocal;
    }
    else
    {
        GMX_THROW(
                gmx::InconsistentInputError("Only Local and NonLocal atom localities can be "
                                            "converted to interaction locality."));
    }
}

} // namespace gmx

#endif // GMX_MDTYPES_LOCALITY_H
