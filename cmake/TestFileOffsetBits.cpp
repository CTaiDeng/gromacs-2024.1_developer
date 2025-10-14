/*
 * Copyright (C) 2025 GaoZheng
 * SPDX-License-Identifier: GPL-3.0-only
 * This file is part of this project.
 * Licensed under the GNU General Public License version 3.
 * See https://www.gnu.org/licenses/gpl-3.0.html for details.
 */
#include <sys/types.h>

int main()
{
  /* Cause a compile-time error if off_t is smaller than 64 bits */
  int off_t_is_large[sizeof(off_t)-7];
  return off_t_is_large[0];
}
