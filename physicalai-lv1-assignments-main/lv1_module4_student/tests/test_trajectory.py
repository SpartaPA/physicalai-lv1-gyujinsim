"""문제 4 — 궤적 보간 검증 (pytest). [완성본]"""

import numpy as np
import pytest

from src.trajectory import cubic_spline_interp, finite_diff, linear_interp, quintic_profile

T_WP = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
Q_WP_1D = np.array([0.0, 0.8, 0.3, 1.2, 0.5, 0.9])
P_WP_3D = np.array([
    [0.20, -0.30, 0.60],
    [0.30, -0.15, 0.75],
    [0.42, 0.00, 0.80],
    [0.50, 0.15, 0.70],
    [0.55, 0.25, 0.55],
    [0.60, 0.30, 0.45],
])


@pytest.fixture(params=["1d", "3d"])
def waypoints(request):
    return (T_WP, Q_WP_1D) if request.param == "1d" else (T_WP, P_WP_3D)


# --- 1. 선형 보간이 경유점을 지나는가 -----------------------------------------

def test_linear_interp_hits_waypoints(waypoints):
    """경유점 시각 t_wp에서 평가했을 때 원래의 q_wp와 일치하는지 검사 (shape 포함)."""
    t_wp, q_wp = waypoints
    q_eval = linear_interp(t_wp, q_wp, t_wp)

    assert q_eval.shape == q_wp.shape
    assert np.allclose(q_eval, q_wp)


# --- 2. 큐빅 스플라인이 경유점을 지나는가 -------------------------------------

def test_cubic_spline_hits_waypoints(waypoints):
    """경유점 시각 t_wp에서 평가했을 때 원래의 q_wp와 일치하는지 검사 (shape 포함)."""
    t_wp, q_wp = waypoints
    q_eval = cubic_spline_interp(t_wp, q_wp, t_wp)

    assert q_eval.shape == q_wp.shape
    assert np.allclose(q_eval, q_wp)


# --- 3. 5차 다항식 경계 조건 ---------------------------------------------------

def test_quintic_boundary_conditions():
    """양끝(t0, tf)에서 q, qd, qdd 가 지정된 경계 조건을 만족하는지 검사."""
    t0, tf = 0.0, 2.0
    q0, qf = 0.0, 1.0
    t = np.linspace(t0, tf, 201)

    q, qd, qdd = quintic_profile(t, t0, tf, q0, qf)

    # 1) 위치 경계 조건 검사
    assert np.isclose(q[0], q0)
    assert np.isclose(q[-1], qf)

    # 2) 속도 경계 조건 검사 (qd = 0)
    assert np.isclose(qd[0], 0.0, atol=1e-7)
    assert np.isclose(qd[-1], 0.0, atol=1e-7)

    # 3) 가속도 경계 조건 검사 (qdd = 0)
    assert np.isclose(qdd[0], 0.0, atol=1e-7)
    assert np.isclose(qdd[-1], 0.0, atol=1e-7)


# --- 추가 테스트 (권장) -----------------------------------------------------

def test_spline_velocity_is_continuous():
    """스플라인의 속도 변화(점프)가 선형 보간의 점프보다 훨씬 작고 부드러운지 검증."""
    t_dense = np.linspace(T_WP[0], T_WP[-1], 501)
    q_lin = linear_interp(T_WP, Q_WP_1D, t_dense)
    q_spl = cubic_spline_interp(T_WP, Q_WP_1D, t_dense)

    v_lin = finite_diff(q_lin, t_dense)
    v_spl = finite_diff(q_spl, t_dense)

    jump_lin = np.max(np.abs(np.diff(v_lin)))
    jump_spl = np.max(np.abs(np.diff(v_spl)))

    # C2 연속인 스플라인의 속도 점프가 C0인 선형 보간보다 확연히 작아야 함
    assert jump_spl < jump_lin * 0.1


def test_quintic_matches_finite_difference():
    """해석적 속도(qd)와 finite_diff(q, t) 수치 미분 결과가 유사한지 검사."""
    t = np.linspace(0.0, 2.0, 1001)
    q, qd, _ = quintic_profile(t, 0.0, 2.0, 0.0, 1.0)
    qd_num = finite_diff(q, t)

    # 수치 미분 오차 감안 atol 설정
    assert np.allclose(qd[1:-1], qd_num[1:-1], atol=1e-3)


def test_quintic_is_monotonic_for_zero_boundary():
    """경계 속도가 0인 기본형 프로파일은 궤적 중간에 역주행 없이 단조 증가해야 함 (qd >= 0)."""
    t = np.linspace(0.0, 2.0, 201)
    _, qd, _ = quintic_profile(t, 0.0, 2.0, 0.0, 1.0)

    assert np.all(qd >= -1e-7)