"""문제 3 — 쿼터니언과 SLERP 검증 (pytest). [완성본]"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation, Slerp

from src.quaternion import lerp_quat, matrix_to_quaternion, quaternion_to_matrix, slerp
from src.rotation import rodrigues, rot_x, rot_y, rot_z

TS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def q_pair():
    """시작·목표 자세 (노트북 3-2 와 같은 값)."""
    R_start = rot_z(np.deg2rad(-30.0)) @ rot_x(np.deg2rad(20.0))
    R_goal = rot_z(np.deg2rad(120.0)) @ rot_y(np.deg2rad(60.0)) @ rot_x(np.deg2rad(-40.0))
    return matrix_to_quaternion(R_start), matrix_to_quaternion(R_goal)


# --- 1. SLERP 결과는 항상 단위 쿼터니언 ---------------------------------------

@pytest.mark.parametrize("t", TS)
def test_slerp_is_unit_norm(q_pair, t):
    """slerp(q0, q1, t)의 노름(크기)이 항상 1인지 검사."""
    q0, q1 = q_pair
    q_t = slerp(q0, q1, t)
    norm = np.linalg.norm(q_t)
    assert np.isclose(norm, 1.0)


# --- 2. t = 0 / 1 에서 시작·목표 자세 -----------------------------------------

def test_slerp_endpoints(q_pair):
    """t=0 일 때 q0, t=1 일 때 q1과 같은 회전을 나타내는지 검사."""
    q0, q1 = q_pair

    # t = 0 검증
    q_start_interp = slerp(q0, q1, 0.0)
    assert np.isclose(np.abs(np.dot(q_start_interp, q0)), 1.0)
    assert np.allclose(quaternion_to_matrix(q_start_interp), quaternion_to_matrix(q0))

    # t = 1 검증
    q_goal_interp = slerp(q0, q1, 1.0)
    assert np.isclose(np.abs(np.dot(q_goal_interp, q1)), 1.0)
    assert np.allclose(quaternion_to_matrix(q_goal_interp), quaternion_to_matrix(q1))


# --- 추가 테스트 (권장) -----------------------------------------------------

def test_matrix_quaternion_roundtrip(rng):
    """무작위 회전 50개 및 특수각: R -> q -> R 이 원래 행렬로 돌아오는가."""
    test_rotations = [
        rodrigues(rng.normal(size=3), rng.uniform(0.0, np.pi))
        for _ in range(50)
    ]
    # 특수 케이스: 단위행렬(0도), 각 축 180도 회전
    test_rotations += [np.eye(3), rot_x(np.pi), rot_y(np.pi), rot_z(np.pi)]

    for R in test_rotations:
        q = matrix_to_quaternion(R)
        R_back = quaternion_to_matrix(q)
        assert np.allclose(R_back, R, atol=1e-7)


def test_slerp_matches_scipy(q_pair):
    """scipy.spatial.transform.Slerp 와 회전행렬 기준으로 일치하는가."""
    q0, q1 = q_pair
    slerp_scipy = Slerp([0.0, 1.0], Rotation.from_quat([q0, q1]))

    for t in TS:
        q_mine = slerp(q0, q1, t)
        R_mine = quaternion_to_matrix(q_mine)
        R_ref = slerp_scipy(t).as_matrix()
        assert np.allclose(R_mine, R_ref, atol=1e-7)


def test_slerp_nearly_identical_poses(q_pair):
    """거의 같은 두 자세 및 부호 반전 자세에서 NaN 이 나오지 않고 보간되는가."""
    q0, _ = q_pair
    # 극소 회전 (1e-9 rad)
    q_near = matrix_to_quaternion(quaternion_to_matrix(q0) @ rodrigues([0, 0, 1], 1e-9))

    q_mid_near = slerp(q0, q_near, 0.5)
    assert not np.isnan(q_mid_near).any()
    assert np.isclose(np.linalg.norm(q_mid_near), 1.0)

    # 부호만 반대인 동일 회전 (-q0)
    q_mid_flip = slerp(q0, -q0, 0.5)
    assert not np.isnan(q_mid_flip).any()
    assert np.isclose(np.abs(np.dot(q_mid_flip, q0)), 1.0)


def test_lerp_norm_drops_below_one(q_pair):
    """정규화하지 않은 선형 보간의 중간값(t=0.5)은 크기가 1 보다 작아야 한다."""
    q0, q1 = q_pair
    q_mid_lerp = lerp_quat(q0, q1, 0.5, normalize=False)
    norm = np.linalg.norm(q_mid_lerp)
    assert norm < 1.0 - 1e-4