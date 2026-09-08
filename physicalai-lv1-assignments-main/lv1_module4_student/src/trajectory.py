"""문제 4 — 궤적 보간. (학생 작성용 템플릿 완성본)

경유점(waypoint)을 지나는 궤적을 선형 보간 / 큐빅 스플라인으로 만들고,
시작·끝에서 속도와 가속도가 0 이 되는 5차 다항식 프로파일을 구현한다.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

__all__ = ["linear_interp", "cubic_spline_interp", "quintic_profile", "finite_diff"]


def linear_interp(t_wp, q_wp, t) -> np.ndarray:
    """경유점 사이를 직선으로 잇는 보간. 각 차원마다 `np.interp` 를 사용한다."""
    t_wp = np.asarray(t_wp, dtype=np.float64)
    q_wp = np.asarray(q_wp, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)

    # 1차원 스칼라 궤적: (M,) -> (N,)
    if q_wp.ndim == 1:
        return np.interp(t, t_wp, q_wp)

    # 다차원 궤적: (M, D) -> (N, D)
    D = q_wp.shape[1]
    res = np.empty((len(t), D), dtype=np.float64)
    for d in range(D):
        res[:, d] = np.interp(t, t_wp, q_wp[:, d])
    return res


def cubic_spline_interp(t_wp, q_wp, t, bc_type: str = "natural") -> np.ndarray:
    """경유점을 지나는 큐빅 스플라인 보간 (위치·속도·가속도가 모두 연속, C2).

    bc_type : 양끝 경계 조건. "natural" 또는 "clamped".
    """
    t_wp = np.asarray(t_wp, dtype=np.float64)
    q_wp = np.asarray(q_wp, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)

    # clamped의 경우 양끝 1차 도함수(속도)가 0인 경계 조건
    if bc_type == "clamped":
        if q_wp.ndim == 1:
            bc = ((1, 0.0), (1, 0.0))
        else:
            D = q_wp.shape[1]
            bc = ((1, np.zeros(D)), (1, np.zeros(D)))
    else:
        bc = bc_type

    cs = CubicSpline(t_wp, q_wp, axis=0, bc_type=bc)
    return cs(t)


def quintic_profile(t, t0: float, tf: float, q0, qf,
                    v0=0.0, vf=0.0, a0=0.0, af=0.0):
    """5차 다항식 궤적 q(t) 와 그 도함수 (q, qd, qdd) 를 돌려준다.

    q0, qf 가 스칼라이면 (N,), (D,) 이면 (N, D) 를 반환한다.
    """
    t = np.asarray(t, dtype=np.float64)
    q0 = np.asarray(q0, dtype=np.float64)
    qf = np.asarray(qf, dtype=np.float64)
    is_scalar = (q0.ndim == 0)

    # 차원 맞춤: 1D 배열 (D,) 형태로 변환 처리
    q0_arr = np.atleast_1d(q0)
    qf_arr = np.atleast_1d(qf)
    D = q0_arr.shape[0]

    v0_arr = np.broadcast_to(np.asarray(v0, dtype=np.float64), (D,))
    vf_arr = np.broadcast_to(np.asarray(vf, dtype=np.float64), (D,))
    a0_arr = np.broadcast_to(np.asarray(a0, dtype=np.float64), (D,))
    af_arr = np.broadcast_to(np.asarray(af, dtype=np.float64), (D,))

    T = float(tf - t0)
    tau = (t - t0) / T  # 정규화 시간 (0 ~ 1)

    # 일반적인 5차 다항식 경계조건 계수 계산 (시간 스케일링 형태)
    # q(tau) = a0_c + a1_c*tau + a2_c*tau^2 + a3_c*tau^3 + a4_c*tau^4 + a5_c*tau^5
    # (여기서 도함수는 d/dt = (1/T) d/dtau)
    delta_q = qf_arr - q0_arr
    v0_scaled = v0_arr * T
    vf_scaled = vf_arr * T
    a0_scaled = a0_arr * (T ** 2)
    af_scaled = af_arr * (T ** 2)

    c0 = q0_arr
    c1 = v0_scaled
    c2 = 0.5 * a0_scaled
    c3 = 10.0 * delta_q - (6.0 * v0_scaled + 4.0 * vf_scaled) + (0.5 * af_scaled - 1.5 * a0_scaled)
    c4 = -15.0 * delta_q + (8.0 * v0_scaled + 7.0 * vf_scaled) + (1.5 * a0_scaled - af_scaled)
    c5 = 6.0 * delta_q - 3.0 * (v0_scaled + vf_scaled) + 0.5 * (af_scaled - a0_scaled)

    # tau 차수별 배열: shape (N, 1)
    tau_col = tau[:, np.newaxis]
    tau2 = tau_col ** 2
    tau3 = tau_col ** 3
    tau4 = tau_col ** 4
    tau5 = tau_col ** 5

    # 위치 q(t): (N, D)
    q = c0 + tau_col * c1 + tau2 * c2 + tau3 * c3 + tau4 * c4 + tau5 * c5

    # 속도 qd(t) = (1 / T) * dq/dtau: (N, D)
    dq_dtau = c1 + 2.0 * tau_col * c2 + 3.0 * tau2 * c3 + 4.0 * tau3 * c4 + 5.0 * tau4 * c5
    qd = dq_dtau / T

    # 가속도 qdd(t) = (1 / T^2) * d^2q/dtau^2: (N, D)
    d2q_dtau2 = 2.0 * c2 + 6.0 * tau_col * c3 + 12.0 * tau2 * c4 + 20.0 * tau3 * c5
    qdd = d2q_dtau2 / (T ** 2)

    # 스칼라 입력이었다면 (N,) 형태로 축소 반환
    if is_scalar:
        return q.squeeze(axis=-1), qd.squeeze(axis=-1), qdd.squeeze(axis=-1)

    return q, qd, qdd


def finite_diff(y, t) -> np.ndarray:
    """시간축(axis 0)에 대한 수치 미분. `np.gradient(y, t, axis=0)` 를 사용한다."""
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    return np.gradient(y, t, axis=0)