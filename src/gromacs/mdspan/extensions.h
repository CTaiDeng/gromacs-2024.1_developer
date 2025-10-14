/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2019- The GROMACS Authors
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

/*! \libinternal
 * \file
 * \brief GROMACS extensions to mdspan.
 *
 * \author Christian Blau <cblau@gwdg.de>
 * \inlibraryapi
 * \ingroup module_mdspan
 */

#ifndef GMX_MDSPAN_EXTENSIONS_H_
#define GMX_MDSPAN_EXTENSIONS_H_

#include <algorithm>
#include <functional>

#include "gromacs/mdspan/mdspan.h"

namespace gmx
{

/*! \brief
 * Free begin function addressing memory of a contiguously laid out basic_mdspan.
 *
 * \note Changing the elements that basic_mdspan views does not change
 *       the view itself, so a single begin that takes a const view suffices.
 */
template<class BasicMdspan>
constexpr std::enable_if_t<BasicMdspan::is_always_contiguous(), typename BasicMdspan::pointer>
begin(const BasicMdspan& basicMdspan)
{
    return basicMdspan.data();
}

/*! \brief
 * Free end function addressing memory of a contiguously laid out basic_mdspan.
 *
 * \note Changing the elements that basic_mdspan views does not change
 *       the view itself, so a single end that takes a const view suffices.
 */
template<class BasicMdspan>
constexpr std::enable_if_t<BasicMdspan::is_always_contiguous(), typename BasicMdspan::pointer>
end(const BasicMdspan& basicMdspan)
{
    return basicMdspan.data() + basicMdspan.mapping().required_span_size();
}

//! Convenience type for often-used two dimensional extents
using dynamicExtents2D = extents<dynamic_extent, dynamic_extent>;

//! Convenience type for often-used three dimensional extents
using dynamicExtents3D = extents<dynamic_extent, dynamic_extent, dynamic_extent>;

//! Elementwise addition
template<class BasicMdspan>
constexpr BasicMdspan addElementwise(const BasicMdspan& span1, const BasicMdspan& span2)
{
    BasicMdspan result(span1);
    std::transform(begin(span1),
                   end(span1),
                   begin(span2),
                   begin(result),
                   std::plus<typename BasicMdspan::element_type>());
    return result;
}

//! Elementwise subtraction - left minus right
template<class BasicMdspan>
constexpr BasicMdspan subtractElementwise(const BasicMdspan& span1, const BasicMdspan& span2)
{
    BasicMdspan result(span1);
    std::transform(begin(span1),
                   end(span1),
                   begin(span2),
                   begin(result),
                   std::minus<typename BasicMdspan::element_type>());
    return result;
}

//! Elementwise multiplication
template<class BasicMdspan>
constexpr BasicMdspan multiplyElementwise(const BasicMdspan& span1, const BasicMdspan& span2)
{
    BasicMdspan result(span1);
    std::transform(begin(span1),
                   end(span1),
                   begin(span2),
                   begin(result),
                   std::multiplies<typename BasicMdspan::element_type>());
    return result;
}

//! Elementwise division - left / right
template<class BasicMdspan>
constexpr BasicMdspan divideElementwise(const BasicMdspan& span1, const BasicMdspan& span2)
{
    BasicMdspan result(span1);
    std::transform(begin(span1),
                   end(span1),
                   begin(span2),
                   begin(result),
                   std::divides<typename BasicMdspan::element_type>());
    return result;
}

} // namespace gmx

#endif // GMX_MDSPAN_EXTENSIONS_H_
