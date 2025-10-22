# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2010- The GROMACS Authors
# Copyright (C) 2025 GaoZheng
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""LBOPB Discrete SAC module (rlsac).

This package provides a minimal, self-contained discrete SAC implementation
based on PyTorch, with a dummy environment and a training entry point.

Default device is CPU; a configurable GPU toggle is exposed via config.json.
"""

from .env import DummyEnv, SimpleBoxFloat32, SimpleBoxInt32
from .models import DiscretePolicy, QNetwork
from .replay_buffer import ReplayBuffer
from .train import train

__all__ = [
    "DummyEnv",
    "SimpleBoxFloat32",
    "SimpleBoxInt32",
    "DiscretePolicy",
    "QNetwork",
    "ReplayBuffer",
    "train",
]


