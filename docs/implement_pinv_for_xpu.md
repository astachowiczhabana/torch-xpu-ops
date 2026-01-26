# Steps to implement `torch.linalg.pinv` for XPU

## Analysis
- Find `linalg_pinv` implementation in `aten/src/ATen/native/LinearAlgebra.cpp`
- Identify that `linalg_pinv` is a composite operation that doesn't have its own kernel
- Trace the code to find it calls `at::linalg_svd()` for general matrices or `at::linalg_eigh()` for hermitian matrices
- Trace further to find these call internal functions `at::_linalg_svd()` and `at::_linalg_eigh()`
- Check `aten/src/ATen/native/native_functions.yaml` to find these use structured kernels with dispatch stubs (`svd_stub`, `linalg_eigh_stub`)
- Verify these ops are in `third_party/torch-xpu-ops/src/ATen/native/xpu/XPUFallback.template`, confirming they currently fall back to CPU

## Implementation
- Add `_linalg_svd` and `_linalg_svd.U` entries with `XPU: _linalg_svd_out` dispatch to `third_party/torch-xpu-ops/yaml/native/native_functions.yaml`
- Add `_linalg_eigh` and `_linalg_eigh.eigenvalues` entries with `XPU: _linalg_eigh_out` dispatch to `third_party/torch-xpu-ops/yaml/native/native_functions.yaml`
- Remove `_linalg_svd.U` and `_linalg_eigh.eigenvalues` from the fallback list in `third_party/torch-xpu-ops/src/ATen/native/xpu/XPUFallback.template`
- Register `svd_stub` and `linalg_eigh_stub` with `REGISTER_XPU_DISPATCH` in `third_party/torch-xpu-ops/src/ATen/native/xpu/BatchLinearAlgebra.cpp`
- Add `svd_mkl` and `linalg_eigh_mkl` declarations to `third_party/torch-xpu-ops/src/ATen/native/xpu/mkl/BatchLinearAlgebra.h`
- Implement `svd_mkl` and `linalg_eigh_mkl` in `third_party/torch-xpu-ops/src/ATen/native/xpu/mkl/BatchLinearAlgebra.cpp` (or use CPU fallback)
- Re-run cmake configuration to regenerate XPU codegen headers
- Rebuild with ninja
