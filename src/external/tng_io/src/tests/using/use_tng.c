/*
 * Copyright (C) 2025- GaoZheng
 * SPDX-License-Identifier: GPL-3.0-only
 * This file is part of this project.
 * Licensed under the GNU General Public License version 3.
 * See https://www.gnu.org/licenses/gpl-3.0.html for details.
 */
#include "tng/tng_io.h"

void test_tng(void)
{
    tng_trajectory_t data;
    char             buf[256];
    tng_version(data, buf, 256);
}
