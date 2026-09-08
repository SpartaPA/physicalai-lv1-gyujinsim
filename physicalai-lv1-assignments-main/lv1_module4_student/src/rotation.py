"""문제 2·3 — 회전 행렬 모듈. (학생 작성용 템플릿)

축별 회전 행렬, 로드리게스 공식(임의 축 회전), Gram-Schmidt 재직교화,
회전행렬 판정과 고유값 분해 기반 축·각 복원을 직접 구현한다.

문제 1 에서 만든 `src/vectors.py` 를 그대로 재사용한다.
"""

from __future__ import annotations

import numpy as np

from .vectors import det, normalize, skew

__all__ = [
    "rot_x",
    "rot_y",
    "rot_z",
    "rodrigues",
    "gram_schmidt",
    "orthogonality_error",
    "is_rotation",
    "axis_angle_from_matrix",
    "quaternion_from_axis_angle",
]


# ------------------------------------------------------------ 축별 회전 행렬

def rot_x(theta: float) -> np.ndarray:
    """x축 기준 회전 행렬 (theta 는 **라디안**). x 성분은 보존된다."""
    # TODO: 문제 2-1
    result = np.array([
        [1,0,0],
        [0,np.cos(theta),-np.sin(theta)],
        [0,np.sin(theta),np.cos(theta)]
    ])

    return result


def rot_y(theta: float) -> np.ndarray:
    """y축 기준 회전 행렬 (theta 는 라디안). y 성분은 보존된다.

    부호 배치가 x·z 와 반대로 보이는 이유는 노트북 2-1 에서 설명한다.
    """
    # TODO: 문제 2-1
    result = np.array([
        [np.cos(theta),0,np.sin(theta)],
        [0,1,0],
        [-np.sin(theta),0,np.cos(theta)]
    ])

    return result


def rot_z(theta: float) -> np.ndarray:
    """z축 기준 회전 행렬 (theta 는 라디안). z 성분은 보존된다."""
    # TODO: 문제 2-1
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([[c, -s, 0.0],
                     [s, c, 0.0],
                     [0.0, 0.0, 1.0]
    ],dtype=float)


def rodrigues(axis, theta: float) -> np.ndarray:
    """로드리게스 공식으로 임의 축 회전 행렬을 만든다.

        R = I + sin(theta) * K + (1 - cos(theta)) * K @ K,   K = [k]_x

    - 축은 함수 안에서 단위벡터로 정규화한다
      (정규화되지 않은 축을 넣어도 같은 결과가 나와야 한다).
    - 문제 1 의 `skew` 를 반드시 사용한다.
    """
    # TODO: 문제 2-5
    axis = np.array(axis)

    n_axis = normalize(axis)
    kx,ky,kz = n_axis

    k = skew(n_axis)
    k2 = np.array([
        [-((kz**2)+(ky**2)), kx*ky, kz*kx],
        [kx*ky, -((kz**2)+(kx**2)), ky*kz],
        [kx*kz, ky*kz, -((ky**2) +(kx**2))]
    ])
    
    return np.eye(3) + np.sin(theta) * k + (1 - np.cos(theta)) * k2


# ------------------------------------------------------------- 재직교화 관련

def gram_schmidt(A) -> np.ndarray:
    """**열벡터**에 대해 Gram-Schmidt 직교정규화를 수행한다.

        q1 = a1 / |a1|
        vj = aj - sum_{i<j} (qi · aj) qi
        qj = vj / |vj|

    각 열에서 앞선 열 방향 성분(정사영)을 빼고 정규화하는 것이며,
    문제 1 의 project / reject 와 같은 연산의 반복이다.

    수치적으로는 성분을 빼자마자 갱신하는 modified Gram-Schmidt 가 더 안정적이다.
    앞선 열들에 종속인 열이 있으면 ValueError.
    """
    # TODO: 문제 3-2
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("정사각 행렬이어야 합니다.")

    Q = A.copy()
    n = Q.shape[1]

    for j in range(n):
        # 앞서 구한 직교 기저 성분들을 제거 (Modified Gram-Schmidt)
        for i in range(j):
            proj = np.dot(Q[:, j], Q[:, i]) * Q[:, i]
            Q[:, j] -= proj

        length = np.linalg.norm(Q[:, j])
        if length <= 1e-10:
            raise ValueError("열벡터들이 선형종속입니다.")

        Q[:, j] /= length

    return Q


def orthogonality_error(R) -> float:
    """직교성 이탈 지표: || R^T R - I ||_F  (프로베니우스 노름).

    완전한 직교행렬이면 0 이고, 클수록 직교성이 무너진 것이다.
    """
    # TODO: 문제 3-1
    R = np.asarray(R, dtype=float)
    E = R.T @ R - np.eye(R.shape[0])
    return float(np.sqrt(np.sum(E * E)))

def is_rotation(R, atol: float = 1e-8) -> bool:
    """회전행렬 판정: 직교(R^T R = I) **그리고** det(R) = +1 이면 True.

    det = -1 이면 직교이긴 하지만 반사가 섞여 있어 회전이 아니다.
    3x3 이 아니면 False.
    """
    # TODO: 문제 3-2
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        return False

    err = orthogonality_error(R)
    det_val = det(R)

    # 넘겨받은 atol 기준으로 판정
    return (err < atol) and np.isclose(det_val, 1.0, atol=atol)

# --------------------------------------------------- 회전축·회전각·쿼터니언

def axis_angle_from_matrix(R, atol: float = 1e-8):
    """고유값 분해로 회전축을, 대각합으로 회전각을 복원한다.

    - 회전축은 고유값 1 에 대응하는 실수 고유벡터다 (R k = k).
      -> 여기서는 `np.linalg.eig` 를 써도 된다 (검산이 아니라 축 복원이 목적).
    - 회전각은 trace(R) = 1 + 2 cos(theta) 에서 구한다.
    - arccos 의 치역이 [0, pi] 라 '어느 쪽으로 도는지'는 알 수 없고,
      고유벡터도 부호가 정해지지 않는다. 반대칭 성분
      R - R^T = 2 sin(theta) [k]_x 를 이용해 부호를 맞춘다.
    - theta = 0 (회전 없음) 과 theta = pi (sin = 0) 는 따로 처리해야 한다.
      두 경우에 어떤 규약을 쓸지 정하고 주석으로 남긴다.

    Returns
    -------
    axis : 단위 회전축 (3,)
    angle : 회전각 [rad], 0 <= angle <= pi
    """
    # TODO: 문제 6-4
    R = np.asarray(R, dtype=float)

    # 1. 회전각 복원: trace(R) = 1 + 2 cos(theta)
    tr = np.trace(R)
    cos_theta = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cos_theta))

    # 2-1. 특이 케이스 1: theta ≈ 0 (회전 없음)
    # [규약] 회전각이 0이면 물리적으로 고정된 축이 없으므로, 기본 단위 벡터 [0, 0, 1]을 축으로 규약한다.
    if np.isclose(angle, 0.0, atol=atol):
        return np.array([0.0, 0.0, 1.0]), 0.0

    # 2-2. 특이 케이스 2: theta ≈ pi (180도 회전, sin(theta) ≈ 0)
    # [규약] R - R.T ≈ 0이 되어 반대칭 행렬로 축을 구할 수 없으므로,
    # np.linalg.eig를 통해 고유값 1에 대응하는 고유벡터를 축으로 사용한다.
    # (180도 회전은 축의 부호 +k, -k가 동일한 물리적 회전 상태를 나타냄)
    if np.isclose(angle, np.pi, atol=atol):
        eigenvalues, eigenvectors = np.linalg.eig(R)
        # 고유값 중 1에 가장 가까운 인덱스 탐색
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        axis = np.real(eigenvectors[:, idx])
        axis = axis / np.linalg.norm(axis)
        return axis, float(np.pi)

    # 2-3. 일반적인 경우 (0 < theta < pi)
    # R - R.T = 2 * sin(theta) * [k]_x
    # sin(theta) > 0 이므로 반대칭 성분에서 바로 축 벡터의 방향과 부호를 복원할 수 있다.
    rx = R[2, 1] - R[1, 2]
    ry = R[0, 2] - R[2, 0]
    rz = R[1, 0] - R[0, 1]

    axis = np.array([rx, ry, rz], dtype=float)
    axis = axis / np.linalg.norm(axis)

    return axis, angle

def quaternion_from_axis_angle(axis, angle: float) -> np.ndarray:
    """축-각에서 단위 쿼터니언을 만든다.

        q = (k * sin(theta/2), cos(theta/2))

    반환 순서는 SciPy `Rotation.as_quat()` 와 같은 **(x, y, z, w)** 로 맞춘다
    (그래야 문제 6-5 에서 바로 비교할 수 있다).
    """
    # TODO: 문제 6-5
    sin_half_angle = np.sin(angle/2)

    front = np.array(sin_half_angle*(axis))
    back = np.cos(angle/2)

    quaternion = np.append(front,back)

    #if quaternion[3] < 0:
        #quaternion = -quaternion

    return quaternion