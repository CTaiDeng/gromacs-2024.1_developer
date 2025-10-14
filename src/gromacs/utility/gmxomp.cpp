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
 * Implements functions from gmxomp.h.
 *
 * \ingroup module_utility
 */
#include "gmxpre.h"

#include "gromacs/utility/gmxomp.h"

#include "config.h"

#include <cstdio>
#include <cstdlib>

#if GMX_OPENMP
#    include <omp.h>
#endif

#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/programcontext.h"
#include "gromacs/utility/stringutil.h"

int gmx_omp_get_max_threads()
{
#if GMX_OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

int gmx_omp_get_num_procs()
{
#if GMX_OPENMP
    return omp_get_num_procs();
#else
    return 1;
#endif
}

int gmx_omp_get_thread_num()
{
#if GMX_OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

void gmx_omp_set_num_threads(int num_threads)
{
#if GMX_OPENMP
    omp_set_num_threads(num_threads);
#else
    GMX_UNUSED_VALUE(num_threads);
#endif
}

bool gmx_omp_check_thread_affinity(char** message)
{
    bool shouldSetAffinity = true;

    *message = nullptr;
#if GMX_OPENMP
    /* We assume that the affinity setting is available on all platforms
     * gcc supports. Even if this is not the case (e.g. Mac OS) the user
     * will only get a warning. */
#    if defined(__GNUC__)
    const char* programName;
    try
    {
        programName = gmx::getProgramContext().displayName();
    }
    GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR

    const char* const gomp_env            = getenv("GOMP_CPU_AFFINITY");
    const bool        bGompCpuAffinitySet = (gomp_env != nullptr);

    /* turn off internal pinning if GOMP_CPU_AFFINITY is set & non-empty */
    if (bGompCpuAffinitySet && *gomp_env != '\0')
    {
        try
        {
            std::string buf = gmx::formatString(
                    "NOTE: GOMP_CPU_AFFINITY set, will turn off %s internal affinity\n"
                    "      setting as the two can conflict and cause performance degradation.\n"
                    "      To keep using the %s internal affinity setting, unset the\n"
                    "      GOMP_CPU_AFFINITY environment variable.",
                    programName,
                    programName);
            *message = gmx_strdup(buf.c_str());
        }
        GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR
        shouldSetAffinity = false;
    }
#    endif /* __GNUC__ */

#endif /* GMX_OPENMP */
    return shouldSetAffinity;
}
