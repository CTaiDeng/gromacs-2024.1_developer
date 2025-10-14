# Copyright (C) 2025 GaoZheng
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of this project.
# Licensed under the GNU General Public License version 3.
# See https://www.gnu.org/licenses/gpl-3.0.html for details.
###########################################################################
#                                                                         #
#    physical_validation,                                                 #
#    a python package to test the physical validity of MD results         #
#                                                                         #
#    Written by Michael R. Shirts <michael.shirts@colorado.edu>           #
#               Pascal T. Merz <pascal.merz@colorado.edu>                 #
#                                                                         #
#    Copyright (C) 2012 University of Virginia                            #
#              (C) 2017 University of Colorado Boulder                    #
#                                                                         #
#    This library is free software; you can redistribute it and/or        #
#    modify it under the terms of the GNU General Public                  #
#    License as published by the Free Software Foundation; either         #
#    the Free Software Foundation, version 3.                             #
#                                                                         #
#    This library is distributed in the hope that it will be useful,      #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of       #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU    #
#    General Public License for more details.                             #
#                                                                         #
#    You should have received a copy of the GNU General Public            #
#    License along with this library; if not, write to the                #
#    Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,     #
#    Boston, MA 02110-1301 USA                                            #
#                                                                         #
###########################################################################
r"""
Physical validation suite for MD simulations

"""

__version__ = "0.1a"

__author__ = "Pascal T. Merz, and Michael R. Shirts"
__copyright__ = "2017"
__credits__ = []
# TODO:
__license__ = "GPL-3.0-only"
__maintainer__ = "Michael R. Shirts"
__email__ = "michael.shirts@colorado.edu"

from . import kinetic_energy
from . import ensemble
from . import integrator
from . import util
from . import data

