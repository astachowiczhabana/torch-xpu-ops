"""
Simple test case for linalg_pinv on XPU.
Run with: python test_pinv_xpu.py
"""
import torch

def test_pinv_xpu():
    # Check XPU availability
    if not torch.xpu.is_available():
        print("XPU is not available!")
        return False

    print(f"XPU device: {torch.xpu.get_device_name(0)}")

    # Create a simple 3x3 matrix on XPU
    A = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 10.0]  # Not singular
    ], device='xpu')

    print(f"Input matrix A:\n{A}")
    print(f"A.device: {A.device}")

    # Compute pseudo-inverse
    A_pinv = torch.linalg.pinv(A)

    print(f"\nPseudo-inverse A_pinv:\n{A_pinv}")
    print(f"A_pinv.device: {A_pinv.device}")

    # Verify: A @ A_pinv @ A should be close to A
    reconstructed = A @ A_pinv @ A
    print(f"\nA @ A_pinv @ A:\n{reconstructed}")

    # Check correctness
    if torch.allclose(A, reconstructed, atol=1e-5):
        print("\n✓ PASSED: A @ A_pinv @ A ≈ A")
    else:
        print("\n✗ FAILED: Reconstruction check failed")
        return False

    return True


def test_pinv_xpu_rectangular():
    """Test with non-square matrix (more common use case for pinv)"""
    if not torch.xpu.is_available():
        return False

    # Tall matrix (overdetermined system)
    A = torch.randn(5, 3, device='xpu')
    A_pinv = torch.linalg.pinv(A)

    print(f"\nRectangular matrix test:")
    print(f"A shape: {A.shape}")
    print(f"A_pinv shape: {A_pinv.shape}")

    # For full column rank: A_pinv @ A should be close to identity
    result = A_pinv @ A
    identity = torch.eye(3, device='xpu')

    if torch.allclose(result, identity, atol=1e-4):
        print("✓ PASSED: A_pinv @ A ≈ I (left inverse)")
    else:
        print("✗ FAILED: Left inverse check failed")
        return False

    return True


def test_pinv_xpu_hermitian():
    """Test with hermitian=True flag (uses eigh instead of svd)"""
    if not torch.xpu.is_available():
        return False

    # Create symmetric positive definite matrix
    A = torch.randn(4, 4, device='xpu')
    A = A @ A.T + torch.eye(4, device='xpu')  # Make it SPD

    print(f"\nHermitian matrix test:")
    A_pinv = torch.linalg.pinv(A, hermitian=True)

    reconstructed = A @ A_pinv @ A
    if torch.allclose(A, reconstructed, atol=1e-4):
        print("✓ PASSED: Hermitian pinv works")
    else:
        print("✗ FAILED: Hermitian pinv failed")
        return False

    return True


if __name__ == "__main__":
    print("=" * 50)
    print("Testing torch.linalg.pinv on XPU")
    print("=" * 50)

    try:
        test_pinv_xpu()
        test_pinv_xpu_rectangular()
        test_pinv_xpu_hermitian()
        print("\n" + "=" * 50)
        print("All tests completed!")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
