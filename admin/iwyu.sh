#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
#
# Copyright (C) 2014- The GROMACS Authors
# Copyright (C) 2025 GaoZheng
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# ---
#
# This file is part of a modified version of the GROMACS molecular simulation package.
# For details on the original project, consult https://www.gromacs.org.
#
# To help fund GROMACS development, we humbly ask that you cite
# the research papers on the package. Check out https://www.gromacs.org.

filename=
build_path=.
src_path=..
cmd="include-what-you-use -DHAVE_CONFIG_H -mavx"

# Read all special arguments and add others to the command
apply=0
for arg in "$@"; do
    if [ $arg == "--apply" ]; then
	apply=1
    elif [[ $arg == -[SB] ]]; then
	echo -S and -B require an argument
	exit 1
    elif [[ $arg == -B* ]]; then
	build_path=${arg:2}
    elif [[ $arg == -S* ]]; then
	src_path=${arg:2}
    elif [[ $arg != -* ]]; then
	if [ "$filename" == "" ]; then
	    filename=$arg
	else
	    echo "This script can only be run on one file at a time"
	    exit 1
	fi
    else
	cmd="$cmd $arg"
    fi
done

if [ "$filename" == "" ]; then
    echo "No file specified"
    exit 1
fi

# We cannot detect whether it is a C++ or C header. Should be fine to always use C++
if [ "${filename##*.}" == "h" ]; then
    cmd="$cmd -x c++"
fi

cmd="$cmd $filename"

# Always use C++11.
if [ "${filename##*.}" == "cpp" -o "${filename##*.}" == "h" ]; then
    cmd="$cmd -std=c++11"
fi

# keep gmxpre.h for source files
if [ "${filename##*.}" == "cpp" -o "${filename##*.}" == "c" ]; then
    cmd="$cmd  -Xiwyu --pch_in_code -Xiwyu --prefix_header_includes=keep"
fi

if [ $src_path == "." ]; then
    src_folder="src" # ./src confuses IWYU
else
    src_folder="$src_path/src"
fi

cmd="$cmd -I${src_folder} -I${src_folder}/external/thread_mpi/include
     -I$build_path/src -I${src_folder}/external/boost
     -Xiwyu --mapping_file=${src_path}/admin/iwyu.imp"

if [ $apply -eq 1 ] ; then
    cmd="$cmd 2>&1 | fix_includes.py --nosafe_headers"
fi

eval $cmd
