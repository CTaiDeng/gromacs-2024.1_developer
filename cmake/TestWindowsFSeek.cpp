/*
 * Copyright (C) 2025 GaoZheng
 * SPDX-License-Identifier: GPL-3.0-only
 * This file is part of this project.
 * Licensed under the GNU General Public License version 3.
 * See https://www.gnu.org/licenses/gpl-3.0.html for details.
 */

#include <stdio.h>
    
int main()
{
  __int64 off=0;

  _fseeki64(NULL, off, SEEK_SET);
        
  return 0;
}
