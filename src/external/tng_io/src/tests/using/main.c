/*
 * Copyright (C) 2025 GaoZheng
 * SPDX-License-Identifier: GPL-3.0-only
 * This file is part of this project.
 * Licensed under the GNU General Public License version 3.
 * See https://www.gnu.org/licenses/gpl-3.0.html for details.
 */
extern void test_tng(void);
extern void test_zlib(void);

int main(int argc, char *argv[])
{
    test_tng();
    test_zlib();
    return 0;
}
