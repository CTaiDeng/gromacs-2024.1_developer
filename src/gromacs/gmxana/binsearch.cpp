/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2010- The GROMACS Authors
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

#include "binsearch.h"

#include "gromacs/utility/real.h"

/*Make range-array (Permutation identity) for sorting */
void rangeArray(int* ar, int size)
{
    int i;
    for (i = 0; i < size; i++)
    {
        ar[i] = i;
    }
}

static void pswap(int* v1, int* v2)
{
    int temp;
    temp = *v1;
    *v1  = *v2;
    *v2  = temp;
}


static void Swap(real* v1, real* v2)
{
    real temp;
    temp = *v1;
    *v1  = *v2;
    *v2  = temp;
}


void insertionSort(real* arr, int* perm, int startndx, int endndx, int direction)
{
    int i, j;

    if (direction >= 0)
    {
        for (i = startndx; i <= endndx; i++)
        {
            j = i;

            while (j > startndx && arr[j - 1] > arr[j])
            {
                Swap(&arr[j], &arr[j - 1]);
                pswap(&perm[j], &perm[j - 1]);
                j--;
            }
        }
    }

    if (direction < 0)
    {
        for (i = startndx; i <= endndx; i++)
        {
            j = i;

            while (j > startndx && arr[j - 1] < arr[j])
            {
                Swap(&arr[j], &arr[j - 1]);
                pswap(&perm[j], &perm[j - 1]);
                j--;
            }
        }
    }
}


int BinarySearch(const real* array, int low, int high, real key, int direction)
{
    int iMid, iMax, iMin;
    iMax = high + 2;
    iMin = low + 1;

    /*Iterative implementation*/

    if (direction >= 0)
    {
        while (iMax - iMin > 1)
        {
            iMid = (iMin + iMax) >> 1;
            if (key < array[iMid - 1])
            {
                iMax = iMid;
            }
            else
            {
                iMin = iMid;
            }
        }
        return iMin;
    }
    else
    {
        while (iMax - iMin > 1)
        {
            iMid = (iMin + iMax) >> 1;
            if (key > array[iMid - 1])
            {
                iMax = iMid;
            }
            else
            {
                iMin = iMid;
            }
        }
        return iMin - 1;
    }
}


int start_binsearch(real* array, int* perm, int low, int high, real key, int direction)
{
    insertionSort(array, perm, low, high, direction);
    return BinarySearch(array, low, high, key, direction);
}
