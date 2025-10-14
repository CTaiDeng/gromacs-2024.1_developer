/*
 * Copyright (C) 2025 GaoZheng
 * SPDX-License-Identifier: GPL-3.0-only
 * This file is part of this project.
 * Licensed under the GNU General Public License version 3.
 * See https://www.gnu.org/licenses/gpl-3.0.html for details.
 */
#define TESTNAME "Coding of velocities. Triplet one-to-one. Cubic cell."
#define FILENAME "test22.tng_compress"
#define ALGOTEST
#define NATOMS 1000
#define CHUNKY 100
#define SCALE 0.1
#define PRECISION 0.01
#define WRITEVEL 1
#define VELPRECISION 0.1
#define INITIALCODING 5
#define INITIALCODINGPARAMETER 0
#define CODING 5
#define CODINGPARAMETER 0
#define INITIALVELCODING 1
#define INITIALVELCODINGPARAMETER -1
#define VELCODING 3
#define VELCODINGPARAMETER -1
#define INTMIN1 0
#define INTMIN2 0
#define INTMIN3 0
#define INTMAX1 10000
#define INTMAX2 10000
#define INTMAX3 10000
#define NFRAMES 1000
#define EXPECTED_FILESIZE 6988699.
