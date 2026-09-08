"""문제 5 — 4x4 동차변환 모듈. (학생 작성용 템플릿)

동차변환 생성/역변환, 점과 방향의 구분, 벡터화된 점군 변환,
정규방정식 기반 최소자승법을 직접 구현한다.
"""

from __future__ import annotations

import numpy as np

from .vectors import inverse_gauss_jordan

__all__ = [
    "make_T",
    "inv_T",
    "inv_T_batch",
    "to_homogeneous",
    "transform_point",
    "transform_direction",
    "transform_points",
    "least_squares_normal_equation",
    "rmse",
]


def make_T(R, t) -> np.ndarray:
    """회전 R(3x3)과 병진 t(3,)로 4x4 동차변환을 만든다.

        T = [[R, t],
             [0, 1]]

    R 이 3x3 이 아니면 ValueError.
    """
    # TODO: 문제 5-1
    R = np.asarray(R,dtype=float)
    if R.shape != (3,3):
        raise ValueError("회전행렬이 3x3 행렬이 아닙니다.")
    T = np.eye(4)
    T[:3,:3] = R
    T[:3, 3] = t
    return T


def inv_T(T) -> np.ndarray:
    """동차변환의 역변환. **일반 역행렬 함수를 쓰지 않고** 공식으로 구한다.

        T^-1 = [[R^T, -R^T t],
                [  0,      1]]

    유도: T^-1 을 [[S, u], [0, 1]] 로 두고 T T^-1 = I 를 풀면
          R S = I -> S = R^T (R 이 직교),  R u + t = 0 -> u = -R^T t.

    4x4 가 아니면 ValueError.
    """
    # TODO: 문제 5-1
    T = np.asarray(T,dtype=float)
    if T.shape != (4,4):
        raise ValueError("받은 행렬이 4x4 행렬이 아닙니다.")
    
    Ti = np.eye(4,dtype=float)

    R = T[:3,:3]
    t = T[:3,3]

    Ti[:3,:3] = R.T
    Ti[:3,3] = -(R.T @ t)

    return Ti


def inv_T_batch(Ts) -> np.ndarray:
    """(N, 4, 4) 동차변환 묶음을 **반복문 없이** 한 번에 역변환한다.

    `inv_T` 와 같은 공식을 배치 축으로 확장한 것이다.
    문제 5-4 의 속도 비교에서 쓴다 — 단건 호출은 파이썬/NumPy 호출 오버헤드가
    지배해서 연산량 차이가 드러나지 않기 때문이다.

    힌트: 전치는 `np.swapaxes(..., 1, 2)`, 배치 행렬-벡터 곱은
          `np.einsum("nij,nj->ni", ...)` 로 쓸 수 있다.
    """
    # TODO: 문제 5-4
    n = Ts.shape[0]
    Ti = np.zeros((n,4,4))

    Ti[:,3,3] = 1

    R_T = Ts[:,:3,:3].transpose(0,2,1)

    t = Ts[:,:3,3:4]

    Ti[:,:3,:3] = R_T
    Ti[:,:3,3:4] = -(R_T@t)

    return Ti


def to_homogeneous(P, w: float = 1.0) -> np.ndarray:
    """(3,) 또는 (N,3) 좌표에 마지막 성분 w 를 붙인다.

    w = 1 이면 점(위치), w = 0 이면 방향(벡터).
    """
    # TODO: 문제 5-2
    P = np.asarray(P)

    if P.ndim==1:
        return np.append(P,w)
    else:
        N = P.shape[0]
        w_col = np.full((N,1),w)
        return np.hstack([P,w_col])


def transform_point(T, p) -> np.ndarray:
    """점 변환 (w = 1): 회전과 병진이 모두 적용된다. 반환은 (3,)."""
    # TODO: 문제 5-2
    R = T[:3,:3]
    t = T[:3,3]
    return R @ p + t


def transform_direction(T, v) -> np.ndarray:
    """방향 변환 (w = 0): 회전만 적용되고 병진은 무시된다. 반환은 (3,)."""
    # TODO: 문제 5-2
    R = T[:3,:3]
    return R @ v


def transform_points(T, P, w: float = 1.0) -> np.ndarray:
    """(N,3) 점군을 **반복문 없이** 한 번에 변환한다. (3,) 입력도 받아야 한다.

    힌트: (T @ P_h.T).T 대신 P_h @ T.T 를 쓰면 전치가 한 번으로 끝나고
          메모리 접근도 행 방향이라 캐시에 유리하다.
    """
    # TODO: 문제 5-2 / 6-2
    P = np.asarray(P, dtype = float)
    R = T[:3,:3]
    t = T[:3,3]

    if P.ndim == 1:
        return R @ P + (w * t)
    # P_cam은 Nx3의 행렬인데 원래 p는 3x1의 열벡터로 이루어진다 그런데 P_cam은 원래 3xN의 벡터로 이루어져야 하는데 Nx3이 되었으니 행벡터가 아래로 이어진 행렬로 이루어져 있다 따라서 P_cam 자체가 p^T와 마찬가지이므로 원래 연산인 Rp + t에서 (Rp)^T가 되어서
    # p^T @ R^T가 되어야 하므로 P_cam @ R.T가 되었다.
    return P @ R.T + (w * t)


def least_squares_normal_equation(A, b):
    """정규방정식 (A^T A) x = A^T b 를 직접 세워 최소자승해를 구한다.

    - (A^T A) 의 역행렬은 문제 4 에서 만든 `inverse_gauss_jordan` 으로 구한다
      (`np.linalg.lstsq` 는 노트북에서 **비교 대상**으로만 쓴다).
    - 근거: 잔차 r = b - A x 가 최소일 때 r 은 A 의 열공간에 수직이므로 A^T r = 0.

    Returns
    -------
    x : 최소자승해
    residual : b - A x
    """
    # TODO: 문제 5-5
    x_hat = inverse_gauss_jordan(A.T @ A) @ A.T @ b
    residual = b - (A @ x_hat)

    return x_hat, residual


def rmse(residual) -> float:
    """잔차의 RMSE = sqrt(mean(r^2))."""
    # TODO: 문제 5-5
    return np.sqrt(np.mean(residual**2))
