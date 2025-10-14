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

#ifndef GMX_TRAJECTORY_ENERGYFRAME_H
#define GMX_TRAJECTORY_ENERGYFRAME_H

#include <map>
#include <string>

#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/real.h"

struct t_enxblock;

struct t_energy
{
    //! The current energy.
    real e;
    //! The running average of the energy
    double eav;
    //! The sum of energies until now.
    double esum;
};

/* The frames that are read/written */
struct t_enxframe
{
    double      t;            /* Timestamp of this frame	                     */
    int64_t     step;         /* MD step	                             */
    int64_t     nsteps;       /* The number of steps between frames            */
    double      dt;           /* The MD time step                              */
    int         nsum;         /* The number of terms for the sums in energyGroupPairTerms      */
    int         nre;          /* Number of energies			     */
    int         e_size;       /* Size (in bytes) of energies		     */
    int         e_alloc;      /* Allocated size (in elements) of energyGroupPairTerms          */
    t_energy*   ener;         /* The energies                                  */
    int         nblock;       /* Number of following energy blocks             */
    t_enxblock* block;        /* The blocks                                    */
    int         nblock_alloc; /* The number of blocks allocated                */
};

namespace gmx
{

/*! \internal
 * \brief Contains the content of an .edr frame read by an EnergyFrameReader
 *
 * The interface of this class is intended to resemble a subset of std::map. */
class EnergyFrame
{
public:
    //! Convenience type
    using MapType = std::map<std::string, real>;
    //! Convenience type
    using MapConstIterator = MapType::const_iterator;
    //! Constructor
    EnergyFrame(const t_enxframe& enxframe, const std::map<std::string, int>& indicesOfEnergyFields);
    /*! \brief Return string that helps users identify this frame, containing time and step number.
     *
     * \throws std::bad_alloc  when out of memory */
    std::string frameName() const;
    /*! \brief Return the value read for energy \c name.
     *
     * \throws APIError  if \c name was not registered with EnergyFileReader. */
    const real& at(const std::string& name) const;
    //! Return const interator to first element of values.
    MapConstIterator begin() const;
    //! Return const interator to past the end element of values.
    MapConstIterator end() const;
    //! Return a const interator to the element with \c key, or end() if not found.
    MapConstIterator find(const std::string& key) const;

private:
    //! Container for energy values, indexed by name
    MapType values_;
    //! Step number read from the .edr file frame
    std::int64_t step_;
    //! Time read from the .edr file frame
    double time_;
};

} // namespace gmx

#endif
