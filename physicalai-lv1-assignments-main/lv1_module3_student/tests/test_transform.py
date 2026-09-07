"""문제 5 — 동차변환 inv_T 검증 (pytest). [학생 작성용 템플릿]

지시문이 요구하는 것은 `inv_T` 검증이지만,
점/방향 구분과 벡터화, 최소자승까지 함께 검증해 두면 이후 문제에서 안전하다.

실행: 프로젝트 루트에서  pytest -v
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.rotation import rot_x, rot_y, rot_z
from src.transform import (
    inv_T,
    least_squares_normal_equation,
    make_T,
    transform_direction,
    transform_point,
    transform_points,
)


@pytest.fixture
def T():
    """테스트에 쓸 대표 동차변환 하나."""
    R = rot_z(0.9) @ rot_y(-0.35) @ rot_x(1.3)
    return make_T(R, [0.35, -0.15, 0.55])


def test_inv_T_gives_identity(T):
    # inv_T(T) @ T 와 T @ inv_T(T) 가 모두 4x4 단위행렬인지 검사
    T_inv = inv_T(T)
    I_4x4 = np.eye(4)
    
    assert_allclose(T_inv @ T, I_4x4, atol=1e-12, err_msg="inv_T(T) @ T is not identity")
    assert_allclose(T @ T_inv, I_4x4, atol=1e-12, err_msg="T @ inv_T(T) is not identity")


def test_inv_T_matches_generic_inverse(T):
    # inv_T(T) 가 np.linalg.inv(T) 와 일치하는지 검사
    T_inv_custom = inv_T(T)
    T_inv_numpy = np.linalg.inv(T)
    
    assert_allclose(T_inv_custom, T_inv_numpy, atol=1e-12)


def test_point_and_direction_differ(T):
    # 같은 벡터를 점(w=1)/방향(w=0)으로 변환하면 결과가 다르고,
    # 그 차이가 정확히 병진 벡터 T[:3, 3] 이며,
    # 방향 변환은 길이를 보존하는지 검사
    v = np.array([1.2, -0.5, 3.1])
    
    p_trans = transform_point(T, v)
    d_trans = transform_direction(T, v)
    
    # 1. 결과가 다름을 검사
    assert not np.allclose(p_trans, d_trans), "Point and direction transforms should not match"
    
    # 2. 차이가 병진 벡터(translation)인지 검사
    translation = T[:3, 3]
    assert_allclose(p_trans - d_trans, translation, atol=1e-12)
    
    # 3. 방향 벡터의 길이 보존 검사
    assert_allclose(np.linalg.norm(d_trans), np.linalg.norm(v), atol=1e-12)


def test_transform_points_is_vectorized(T):
    # (N,3) 점군을 한 번에 변환한 결과가
    # transform_point 를 반복문으로 돌린 결과와 같은지 검사
    np.random.seed(42)
    pts = np.random.rand(10, 3)
    
    res_vectorized = transform_points(T, pts)
    res_loop = np.array([transform_point(T, p) for p in pts])
    
    assert_allclose(res_vectorized, res_loop, atol=1e-12)


def test_roundtrip_through_inverse(T):
    # T 로 보냈다가 inv_T(T) 로 되돌리면 원래 점군이 나오는지 검사
    np.random.seed(42)
    pts = np.random.rand(15, 3)
    
    forward_pts = transform_points(T, pts)
    backward_pts = transform_points(inv_T(T), forward_pts)
    
    assert_allclose(backward_pts, pts, atol=1e-12)


def test_least_squares_matches_lstsq():
    np.random.seed(42)
    
    # 10x3 행렬 A와 임의의 정답 x, 노이즈가 섞인 b 생성
    A = np.random.rand(10, 3)
    x_true = np.array([1.5, -2.0, 0.5])
    noise = np.random.normal(0, 0.1, 10)
    b = A @ x_true + noise
    
    # 함수 반환값이 튜플일 경우 첫 번째 요소(해 x)만 추출
    res_custom = least_squares_normal_equation(A, b)
    x_custom = res_custom[0] if isinstance(res_custom, tuple) else res_custom
    
    x_np, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    # 해 일치 검사
    assert_allclose(x_custom, x_np, atol=1e-10)
    
    # 잔차(residual) 벡터 r = b - A @ x 계산
    r = b - A @ x_custom
    
    # A^T r = 0 (잔차가 A의 열공간에 직교하는지 검증)
    assert_allclose(A.T @ r, np.zeros(3), atol=1e-10)