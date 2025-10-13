#!/usr/bin/env bash
# Copyright (C) 2025- GaoZheng
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of this project.
# Licensed under the GNU General Public License version 3.
# See https://www.gnu.org/licenses/gpl-3.0.html for details.
#SBATCH -N 12
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH -t 00:60:00
#SBATCH -p parallel
#SBATCH -A kas_dev

# Run parallel program over Infiniband using OpenMPI
module load gcc
module load openmpi

srun python -m mpi4py restrained-ensemble.py 12
