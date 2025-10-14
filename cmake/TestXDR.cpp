/*
 * Copyright (C) 2025 GaoZheng
 * SPDX-License-Identifier: GPL-3.0-only
 * This file is part of this project.
 * Licensed under the GNU General Public License version 3.
 * See https://www.gnu.org/licenses/gpl-3.0.html for details.
 */
#include<rpc/rpc.h>
#include<rpc/xdr.h>

int
main()
{
  /* This should only compile, not run, so set xd to NULL */
  XDR *xd = NULL;
  float f; 
  xdr_float(xd,&f);
  return 0;
}    
