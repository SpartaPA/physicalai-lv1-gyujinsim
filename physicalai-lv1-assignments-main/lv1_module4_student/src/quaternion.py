"""문제 3 — 쿼터니언 변환과 SLERP. (학생 작성용 템플릿 완성본)

규약
----
- 쿼터니언은 길이 4 배열 (x, y, z, w) 다. 모듈 ③ 의 quaternion_from_axis_angle 과
  SciPy Rotation.as_quat() 와 같은 순서다. w 가 스칼라(실수부)다.
- q 와 -q 는 같은 회전이다 (이중 덮개). 비교할 때는 부호를 무시하거나 |q . q_ref| 를 본다.
- 보간은 항상 짧은 호를 택한다: q0 . q1 < 0 이면 q1 의 부호를 뒤집고 시작한다.

SciPy 는 검산(비교) 용도로만 쓴다. 이 파일 안에서는 numpy 만 사용한다.
"""

from __future__ import annotations

import numpy as np

__all__ = ["matrix_to_quaternion", "quaternion_to_matrix", "slerp", "lerp_quat", "quat_angle"]


def matrix_to_quaternion(R) -> np.ndarray:
    """회전행렬 (3,3) -> 단위 쿼터니언 (x, y, z, w).

    Shepperd 방법을 사용하여 trace 및 대각 성분 중 최댓값을 기준으로 
    0 나눗셈을 방지하며 수치적 안정성을 확보합니다.
    반환값은 정규화되며 w >= 0 을 만족합니다.
    """
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)

    if tr > 0.0:
        s = 2.0 * np.sqrt(1.0 + tr)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([x, y, z, w], dtype=np.float64)
    q /= np.linalg.norm(q)

    # w >= 0 규약 적용
    if q[3] < 0.0:
        q = -q

    return q


def quaternion_to_matrix(q) -> np.ndarray:
    """단위 쿼터니언 (x, y, z, w) -> 회전행렬 (3,3)."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    x, y, z, w = q

    R = np.array([
        [1.0 - 2.0 * (y**2 + z**2), 2.0 * (x * y - z * w),       2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w),       1.0 - 2.0 * (x**2 + z**2), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w),       2.0 * (y * z + x * w),       1.0 - 2.0 * (x**2 + y**2)]
    ], dtype=np.float64)

    return R


def quat_angle(q0, q1) -> float:
    """두 단위 쿼터니언이 나타내는 회전 사이의 각도 [rad], 0 <= angle <= pi."""
    q0 = np.asarray(q0, dtype=np.float64) / np.linalg.norm(q0)
    q1 = np.asarray(q1, dtype=np.float64) / np.linalg.norm(q1)

    dot = np.abs(np.dot(q0, q1))
    dot = np.clip(dot, -1.0, 1.0)

    return float(2.0 * np.arccos(dot))


def slerp(q0, q1, t: float, eps: float = 1e-8) -> np.ndarray:
    """구면 선형 보간 (Spherical Linear intERPolation)."""
    q0 = np.asarray(q0, dtype=np.float64) / np.linalg.norm(q0)
    q1 = np.asarray(q1, dtype=np.float64) / np.linalg.norm(q1)

    # 내적 계산
    dot = float(np.dot(q0, q1))

    # 짧은 호 선택: q0 . q1 < 0 이면 q1 부호 반전
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    # 부동소수점 오차 방지
    dot = np.clip(dot, -1.0, 1.0)

    # 두 자세가 거의 평행한 경우 (0 나눗셈 방지 -> 선형 보간 후 정규화)
    if dot > 1.0 - eps:
        res = (1.0 - t) * q0 + t * q1
        return res / np.linalg.norm(res)

    # SLERP 수식 적용
    omega = np.arccos(dot)
    sin_omega = np.sin(omega)

    scale0 = np.sin((1.0 - t) * omega) / sin_omega
    scale1 = np.sin(t * omega) / sin_omega

    q_t = scale0 * q0 + scale1 * q1
    return q_t / np.linalg.norm(q_t)


def lerp_quat(q0, q1, t: float, normalize: bool = False) -> np.ndarray:
    """성분별 단순 선형 보간 (비교용)."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)

    # 짧은 호 선택
    if np.dot(q0, q1) < 0.0:
        q1 = -q1

    q_t = (1.0 - t) * q0 + t * q1

    if normalize:
        q_t = q_t / np.linalg.norm(q_t)

    return q_t