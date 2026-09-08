from __future__ import annotations

import numpy as np

__all__ = ["pca_axes", "kabsch", "fit_plane_lstsq", "remove_outliers", "rotation_angle_deg"]


def rotation_angle_deg(R_a, R_b) -> float:
    """두 회전행렬 사이의 각도 [deg] — R_a^T R_b 의 회전각. (제공 코드)"""
    R_rel = np.asarray(R_a, dtype=float).T @ np.asarray(R_b, dtype=float)
    cos_theta = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cos_theta)))


def pca_axes(P):
    """점군 (N,3) 의 주축을 고유분해로 뽑는다."""
    P = np.asarray(P, dtype=float)
    N = P.shape[0]

    # 1. 중심 계산 및 중심 이동
    centroid = np.mean(P, axis=0)
    X = P - centroid

    # 2. 3x3 공분산 행렬 계산
    C = (X.T @ X) / (N - 1)

    # 3. 공분산 행렬 고유분해 (오름차순 반환됨)
    eigvals, eigvecs = np.linalg.eigh(C)

    # 4. 고유값 내림차순 정렬 및 열벡터 정렬
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    axes = eigvecs[:, idx]

    # 5. det(axes) = +1 (오른손 좌표계) 보정
    if np.linalg.det(axes) < 0:
        axes[:, 2] = -axes[:, 2]

    return axes, eigvals, centroid


def kabsch(P, Q):
    """대응이 알려진 두 점군 P, Q (N,3) 에 대해 Q ~ P @ R.T + t 를 만족하는 (R, t) 를 구한다."""
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)

    # 1. 중심 계산 및 중심 이동
    cP = np.mean(P, axis=0)
    cQ = np.mean(Q, axis=0)
    X = P - cP
    Y = Q - cQ

    # 2. 교차 공분산 행렬 계산
    H = X.T @ Y

    # 3. SVD 분해
    U, _, Vt = np.linalg.svd(H)
    V = Vt.T

    # 4. 반사(Reflection) 방지 보정 행렬 D 구성
    d = np.sign(np.linalg.det(V @ U.T))
    D = np.diag([1.0, 1.0, d])

    # 5. 회전행렬 R 및 병진벡터 t 계산
    R = V @ D @ U.T
    t = cQ - R @ cP

    return R, t


def fit_plane_lstsq(P):
    """점군 (N,3) 에 평면 n . p + d = 0 을 최소제곱으로 피팅한다."""
    P = np.asarray(P, dtype=float)
    x = P[:, 0]
    y = P[:, 1]
    z = P[:, 2]

    # z = a*x + b*y + c 모델: [x, y, 1] @ [a, b, c].T = z
    A = np.column_stack([x, y, np.ones_like(x)])
    
    # 특이 행렬에 안전하도록 lstsq 사용
    (a, b, c), residuals_sum, rank, s = np.linalg.lstsq(A, z, rcond=None)

    # a*x + b*y - z + c = 0 형태이므로 비정규화 법선은 (a, b, -1)
    unnormalized_normal = np.array([a, b, -1.0], dtype=float)
    norm = np.linalg.norm(unnormalized_normal)

    normal = unnormalized_normal / norm
    d = float(c / norm)

    # 각 점의 평면까지 부호 있는 거리: n . p + d
    residuals = P @ normal + d

    return normal, d, residuals


def remove_outliers(P, residuals, k: float = 3.0):
    """잔차가 큰 점을 MAD 기반 이상치 탐지로 제거한다."""
    P = np.asarray(P)
    residuals = np.asarray(residuals, dtype=float)

    # MAD(Median Absolute Deviation) 계산
    med = np.median(residuals)
    mad = np.median(np.abs(residuals - med))
    sigma = 1.4826 * mad

    # 분산이 0에 수렴하는 경우 방지
    if sigma < 1e-12:
        mask = np.ones(len(residuals), dtype=bool)
    else:
        mask = np.abs(residuals) < (k * sigma)

    P_clean = P[mask]

    return P_clean, mask