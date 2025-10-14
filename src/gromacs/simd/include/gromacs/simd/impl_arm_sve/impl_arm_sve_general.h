/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2020- The GROMACS Authors
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

/*
 * armv8+sve support to GROMACS was contributed by the Research Organization for
 * Information Science and Technology (RIST).
 * Copyright (c) 2020 Research Organization for Information Science and Technology (RIST).
 */

#ifndef GMX_SIMD_IMPL_ARM_SVE_GENERAL_H
#define GMX_SIMD_IMPL_ARM_SVE_GENERAL_H

namespace gmx
{

static inline void simdPrefetch(void* m)
{
#ifdef __GNUC__
    __builtin_prefetch(m);
#endif
}

#define SVE_SIMD3_DOUBLE_MASK svwhilelt_b64(0, 3)
#define SVE_SIMD4_DOUBLE_MASK svwhilelt_b64(0, 4)
#define SVE_DOUBLE_MASK svptrue_b64()
#define SVE_DINT32_MASK svptrue_b64()
#define SVE_SIMD_FLOAT_HALF_DOUBLE_MASK svwhilelt_b32(0, (int32_t)GMX_SIMD_DINT32_WIDTH)
#define SVE_SIMD_DOUBLE_HALF_MASK svwhilelt_b64(0, (int32_t)GMX_SIMD_DOUBLE_WIDTH / 2)
#define SVE_FLOAT_HALF_MASK svwhilelt_b32(0, GMX_SIMD_FLOAT_WIDTH / 2)
#define SVE_FINT32_HALF_MASK svwhilelt_b32(0, GMX_SIMD_FLOAT_WIDTH / 2)
#define SVE_FLOAT4_MASK svptrue_pat_b32(SV_VL4)
#define SVE_FLOAT3_MASK svptrue_pat_b32(SV_VL3)
} // namespace gmx

#endif // GMX_SIMD_IMPL_ARM_SVE_GENERAL_H
