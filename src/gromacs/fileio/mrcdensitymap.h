/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2019- The GROMACS Authors
 * Copyright (C) 2025- GaoZheng
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

/*! \libinternal \file
 * \brief
 * Declars mrc/ccp4-file format handling.
 *
 * \author Christian Blau <blau@kth.se>
 *
 * \inlibraryapi
 * \ingroup module_fileio
 */
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "gromacs/math/multidimarray.h"
#include "gromacs/mdspan/extensions.h"

namespace gmx
{
template<typename>
class ArrayRef;
struct MrcDensityMapHeader;
class ISerializer;
class TranslateAndScale;

/*! \libinternal \brief Read an mrc/ccp4 file that contains float values.
 */
class MrcDensityMapOfFloatReader
{
public:
    /*! \brief Construct from directly de-serializing data into the object.
     * \throws InternalError if serializer is not reading
     * \throws InternalError if header is inconsistent
     * \throws if serializer throws error upon failed reading
     * \param[in] serializer Serializer to read the object data from
     */
    explicit MrcDensityMapOfFloatReader(ISerializer* serializer);

    ~MrcDensityMapOfFloatReader();

    //! Return a view on the data of the density grid
    ArrayRef<const float> constView() const;
    //! Return the header
    const MrcDensityMapHeader& header() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/*! \libinternal \brief Read an mrc density map from a given file.
 *
 * Higher level class than MrcDensityMapOfFloatReader that takes a file name
 * upon construction and returns coordinate transformation into the density
 * lattice as well as the density data.
 *
 * Attempts reading with swapped endianess if header is not sane.
 *
 * Performs basic sanity checks on header information and data size.
 *
 * \note File reading is completed during construction. When the constructor
 *       completes succesfully, transformation to density lattice and density
 *       data are valid, irrespective of the state of the read file.
 */
class MrcDensityMapOfFloatFromFileReader
{
public:
    MrcDensityMapOfFloatFromFileReader();

    /*! \brief Read from filename.
     * \throws FileIOError if file does not exist
     * \throws FileIOError if read in buffer size does not match file size
     * \throws FileIOError if header information does not match density
     *                     data size
     */
    explicit MrcDensityMapOfFloatFromFileReader(const std::filesystem::path& filename);

    ~MrcDensityMapOfFloatFromFileReader();

    //! Return the coordinate transformation into the density
    TranslateAndScale transformationToDensityLattice() const;

    //! Return a copy of the density data
    MultiDimArray<std::vector<float>, dynamicExtents3D> densityDataCopy() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/*! \libinternal \brief Write an mrc/ccp4 file that contains float values.
 */
class MrcDensityMapOfFloatWriter
{
public:
    /*! \brief Construct by setting the data and the header.
     *
     * \throws if the header data description does not match the provided data
     *
     * \param[in] header mrc density map header
     * \param[in] data the density map data
     */
    MrcDensityMapOfFloatWriter(const MrcDensityMapHeader& header, ArrayRef<const float> data);

    ~MrcDensityMapOfFloatWriter();

    //! Serialize the mrc density data.
    void write(ISerializer* serializer) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace gmx
