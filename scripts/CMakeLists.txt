# SPDX-License-Identifier: GPL-3.0-only
#
# Copyright (C) 2010- The GROMACS Authors
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

if(APPLE)
  set(LD_LIBRARY_PATH "DYLD_LIBRARY_PATH")
else()
  set(LD_LIBRARY_PATH "LD_LIBRARY_PATH")
endif()
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/GMXRC.cmakein      ${CMAKE_CURRENT_BINARY_DIR}/GMXRC @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/GMXRC.bash.cmakein ${CMAKE_CURRENT_BINARY_DIR}/GMXRC.bash @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/GMXRC.csh.cmakein  ${CMAKE_CURRENT_BINARY_DIR}/GMXRC.csh @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/GMXRC.zsh.cmakein  ${CMAKE_CURRENT_BINARY_DIR}/GMXRC.zsh @ONLY)

install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/GMXRC      DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT runtime)
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/GMXRC.bash DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT runtime)
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/GMXRC.zsh  DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT runtime)
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/GMXRC.csh  DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT runtime)

install(PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/demux.pl      DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT runtime)
install(PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/xplor2gmx.pl  DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT runtime)
