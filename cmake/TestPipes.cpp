/*
 * Copyright (C) 2025 GaoZheng
 * SPDX-License-Identifier: GPL-3.0-only
 * This file is part of this project.
 * Licensed under the GNU General Public License version 3.
 * See https://www.gnu.org/licenses/gpl-3.0.html for details.
 */
#ifdef __CYGWIN__
    /* Pipes need POSIX things, not just std ones */
    #define _POSIX_C_SOURCE 200809L
#endif
#include <stdio.h>

int
main()
{
  FILE *fp;

  fp = popen("/tmp/xyz","r");
  return (fp==NULL);
}
