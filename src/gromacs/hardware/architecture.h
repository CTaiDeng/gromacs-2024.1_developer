/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2017- The GROMACS Authors
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
 * Declares and defines architecture booleans to minimize preprocessed code
 *
 * \ingroup module_hardware
 */
#ifndef GMX_ARCHITECTURE_H
#define GMX_ARCHITECTURE_H

namespace gmx
{

//! Enum for GROMACS CPU hardware detection support
enum class Architecture
{
    Unknown, //!< Not one of the cases below
    X86,     //!< X86
    Arm,     //!< ARM
    PowerPC, //!< IBM PowerPC
    RiscV32, //!< 32-bit RISC-V
    RiscV64  //!< 64-bit RISC-V
};

//! Whether the compilation is targeting 32-bit x86.
#if (defined __i386__ || defined __i386 || defined _X86_ || defined _M_IX86)
#    define GMX_IS_X86_32 1
#else
#    define GMX_IS_X86_32 0
#endif

//! Whether the compilation is targeting 64-bit x86.
#if (defined __x86_64__ || defined __x86_64 || defined __amd64__ || defined __amd64 \
     || defined _M_X64 || defined _M_AMD64)
#    define GMX_IS_X86_64 1
#else
#    define GMX_IS_X86_64 0
#endif

//! Constant that tells what the architecture is
static constexpr Architecture c_architecture =
#if GMX_IS_X86_32 || GMX_IS_X86_64
        Architecture::X86;
#elif defined __arm__ || defined __arm || defined _M_ARM || defined __aarch64__
        Architecture::Arm;
#elif defined __powerpc__ || defined __ppc__ || defined __PPC__
        Architecture::PowerPC;
#elif defined __riscv && defined __riscv_xlen && (__riscv_xlen == 32)
        Architecture::RiscV32;
#elif defined __riscv && defined __riscv_xlen && (__riscv_xlen == 64)
        Architecture::RiscV64;
#else
        Architecture::Unknown;
#endif

} // namespace gmx

#endif // GMX_ARCHITECTURE_H
