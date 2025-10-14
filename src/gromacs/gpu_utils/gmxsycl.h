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

/*! \libinternal \file
 * \brief
 * Wraps the complexity of including SYCL in GROMACS.
 *
 * \inlibraryapi
 */

#ifndef GMX_GPU_UTILS_GMXSYCL_H
#define GMX_GPU_UTILS_GMXSYCL_H

#include "config.h"

// For hipSYCL, we need to activate floating-point atomics
#if GMX_SYCL_HIPSYCL
#    define HIPSYCL_EXT_FP_ATOMICS
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wunused-variable"
#    pragma clang diagnostic ignored "-Wunused-parameter"
#    pragma clang diagnostic ignored "-Wmissing-noreturn"
#    pragma clang diagnostic ignored "-Wshadow-field"
#    pragma clang diagnostic ignored "-Wctad-maybe-unsupported"
#    pragma clang diagnostic ignored "-Wdeprecated-copy-dtor"
#    pragma clang diagnostic ignored "-Winconsistent-missing-destructor-override"
#    pragma clang diagnostic ignored "-Wunused-template"
#    pragma clang diagnostic ignored "-Wsign-compare"
#    pragma clang diagnostic ignored "-Wundefined-reinterpret-cast"
#    pragma clang diagnostic ignored "-Wdeprecated-copy"
#    pragma clang diagnostic ignored "-Wnewline-eof"
#    pragma clang diagnostic ignored "-Wextra-semi"
#    pragma clang diagnostic ignored "-Wsuggest-override"
#    pragma clang diagnostic ignored "-Wsuggest-destructor-override"
#    pragma clang diagnostic ignored "-Wgcc-compat"
#    include <sycl/sycl.hpp>
#    pragma clang diagnostic pop
#else // DPC++
// Needed for CUDA targets https://github.com/intel/llvm/issues/5936, enabled for SPIR automatically
#    if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
#        define SYCL_USE_NATIVE_FP_ATOMICS 1
#    endif
#    include <sycl/sycl.hpp>
#endif

/* Macro to optimize runtime performance by not recording unnecessary events.
 *
 * It relies on the availability of HIPSYCL_EXT_CG_PROPERTY_* extension, and is no-op for
 * other SYCL implementations. Macro can be used as follows (note the lack of comma after it):
 * `queue.submit(GMX_SYCL_DISCARD_EVENT [=](....))`.
 *
 * When this macro is added to `queue.submit`, the returned event should not be used!
 * As a consequence, patterns like `queue.submit(GMX_SYCL_DISCARD_EVENT [=](....)).wait()`
 * must be avoided. If you intend to use the returned event in any way, do not add this macro.
 *
 * The use of the returned event will not necessarily cause run-time errors, but can cause
 * performance degradation (specifically, in hipSYCL the synchronization will be sub-optimal).
 */
#if GMX_SYCL_HIPSYCL
namespace gmx::internal
{
static const sycl::property_list sc_syclDiscardEventProperty_list{
    sycl::property::command_group::hipSYCL_coarse_grained_events()
};
}
#    define GMX_SYCL_DISCARD_EVENT gmx::internal::sc_syclDiscardEventProperty_list,
#else // IntelLLVM does not support command-group properties
#    define GMX_SYCL_DISCARD_EVENT
#endif

#endif
