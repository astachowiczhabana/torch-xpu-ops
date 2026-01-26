/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once

#include <ATen/core/Tensor.h>

namespace at::native::xpu {

TORCH_XPU_API void lu_solve_mkl(
    const Tensor& LU,
    const Tensor& pivots,
    const Tensor& B,
    TransposeType trans);

TORCH_XPU_API void lu_factor_mkl(
    const Tensor& LU,
    const Tensor& pivots,
    const Tensor& info,
    bool pivot);

// SVD decomposition using oneMKL
TORCH_XPU_API void svd_mkl(
    const Tensor& A,
    const bool full_matrices,
    const bool compute_uv,
    const std::optional<std::string_view>& driver,
    const Tensor& U,
    const Tensor& S,
    const Tensor& Vh,
    const Tensor& info);

// Hermitian eigenvalue decomposition using oneMKL
TORCH_XPU_API void linalg_eigh_mkl(
    const Tensor& eigenvalues,
    const Tensor& eigenvectors,
    const Tensor& infos,
    bool upper,
    bool compute_eigenvectors);

} // namespace at::native::xpu
