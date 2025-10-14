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

#include "gmxpre.h"

#include "gromacs/math/utilities.h"

#include "config.h"

#include <cfenv>
#include <cmath>
#include <cstdint>

#include "gromacs/utility/real.h"

#if HAVE_FEDISABLEEXCEPT || (defined(__i386__) || defined(__x86_64__)) && defined(__APPLE__)
//! Floating point exception set that we use and care about
constexpr int c_FPexceptions = FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW;
#endif

bool gmx_within_tol(double f1, double f2, double tol)
{
    /* The or-equal is important - otherwise we return false if f1==f2==0 */
    return fabs(f1 - f2) <= tol * 0.5 * (fabs(f1) + fabs(f2));
}

bool gmx_numzero(double a)
{
    return gmx_within_tol(a, 0.0, GMX_REAL_MIN / GMX_REAL_EPS);
}


bool check_int_multiply_for_overflow(int64_t a, int64_t b, int64_t* result)
{
    int64_t sign = 1;
    if ((0 == a) || (0 == b))
    {
        *result = 0;
        return true;
    }
    if (a < 0)
    {
        a    = -a;
        sign = -sign;
    }
    if (b < 0)
    {
        b    = -b;
        sign = -sign;
    }
    if (INT64_MAX / b < a)
    {
        *result = (sign > 0) ? INT64_MAX : INT64_MIN;
        return false;
    }
    *result = sign * a * b;
    return true;
}

int gmx_feenableexcept()
{
    // While the function is present on RISC-V, actually calling it fails for now
#if HAVE_FEENABLEEXCEPT && !defined(__riscv)
#    if defined(__powerpc__) || defined(__ppc__) || defined(__PPC__)
    feclearexcept(c_FPexceptions);
#    endif
    return feenableexcept(c_FPexceptions);
#elif (defined(__i386__) || defined(__x86_64__)) && defined(__APPLE__)
    /* Author:  David N. Williams
     * License:  Public Domain
     *
     * Might also work on non-Apple Unix. But should be tested
     * before enabling.
     */
    static fenv_t fenv;
    unsigned int  new_excepts = c_FPexceptions & FE_ALL_EXCEPT;

    if (fegetenv(&fenv))
    {
        return -1;
    }

    // unmask
    fenv.__control &= ~new_excepts;
    fenv.__mxcsr &= ~(new_excepts << 7);

    return fesetenv(&fenv);
#else
    return -1;
#endif
}

int gmx_fedisableexcept()
{
    // While the function is present on RISC-V, actually calling it fails for now
#if HAVE_FEDISABLEEXCEPT && !defined(__riscv)
    return fedisableexcept(c_FPexceptions);
#elif (defined(__i386__) || defined(__x86_64__)) && defined(__APPLE__)
    static fenv_t fenv;
    unsigned int  new_excepts = c_FPexceptions & FE_ALL_EXCEPT;
    if (fegetenv(&fenv))
    {
        return -1;
    }

    // mask
    fenv.__control |= new_excepts;
    fenv.__mxcsr |= new_excepts << 7;

    return fesetenv(&fenv);
#else
    return -1;
#endif
}

bool gmxShouldEnableFPExceptions()
{
#if defined(NDEBUG)
    return false; // Release build
#elif defined __clang__ && defined __OPTIMIZE__
    return false; // Buggy compiler
#elif GMX_GPU_SYCL
    return false; // avoid spurious FPE during SYCL JIT
#else
    return true;
#endif
}
