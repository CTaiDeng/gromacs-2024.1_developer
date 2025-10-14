/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Copyright (C) 2018- The GROMACS Authors
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

#include "gmxapi/status.h"

#include <memory>

namespace gmxapi
{

/*! \cond internal
 * \brief Implementation class for Status objects.
 *
 * Current implementation is basically just a wrapper for bool. Future
 * versions should be implemented as a "future" for a result that can contain
 * more information about failures.
 */
class Status::Impl
{
public:
    /*!
     * \brief Default construct as unsuccessful status.
     */
    Impl() : success_{ false } {};

    /*!
     * \brief Construct with success for true input.
     *
     * \param success let Boolean true == success.
     */
    explicit Impl(const bool& success) : success_{ success } {};

    ~Impl() = default;

    /*!
     * \brief Query success status
     *
     * \return true if successful
     */
    bool success() const { return success_; };

private:
    bool success_;
};
/// \endcond

Status::Status() : impl_{ std::make_unique<Status::Impl>() } {}

Status::Status(const Status& status)
{
    impl_ = std::make_unique<Impl>(status.success());
}

Status& Status::operator=(const Status& status)
{
    this->impl_ = std::make_unique<Impl>(status.success());
    return *this;
}

Status& Status::operator=(Status&&) noexcept = default;

Status& Status::operator=(bool success)
{
    this->impl_ = std::make_unique<Impl>(success);
    return *this;
}

Status::Status(Status&&) noexcept = default;

Status::Status(bool success) : impl_{ std::make_unique<Status::Impl>(success) } {}

bool Status::success() const
{
    return impl_->success();
}

Status::~Status() = default;

} // end namespace gmxapi
