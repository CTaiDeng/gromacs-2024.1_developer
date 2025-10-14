# Copyright (C) 2025 GaoZheng
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of this project.
# Licensed under the GNU General Public License version 3.
# See https://www.gnu.org/licenses/gpl-3.0.html for details.
cmake_minimum_required(VERSION 3.2)
project(muparser-example3)

# find muparser target already installed
find_package(muparser 2.0 REQUIRED)

add_executable(example3 example3.cpp)
target_link_libraries(example3 muparser::muparser)

include(CTest)
add_test(example3 example3)

