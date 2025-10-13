# Copyright (C) 2025- GaoZheng
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of this project.
# Licensed under the GNU General Public License version 3.
# See https://www.gnu.org/licenses/gpl-3.0.html for details.
configure_file(Doxyfile.in Doxyfile)

find_package(Doxygen)
if (DOXYGEN_FOUND)
    add_custom_target(docs
                      COMMAND ${DOXYGEN_EXECUTABLE} Doxyfile
                      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                      COMMENT "Generating developer documentation"
                      VERBATIM )
else (DOXYGEN_FOUND)
    message("Doxygen needs to be installed to generate the C++ documentation")
endif (DOXYGEN_FOUND)
