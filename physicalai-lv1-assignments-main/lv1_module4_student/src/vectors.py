"""문제 1 — 벡터 연산 모듈. (학생 작성용 템플릿)

내적 · 사이각 · 정규화 · 정사영 · 반대칭행렬(외적) · 평면 법선과
가우스 소거 기반의 rank / 행렬식 / 역행렬을 **직접** 구현한다.

규칙
----
- `np.linalg` 는 노트북에서 **검산용으로만** 쓰고, 이 모듈 안에서는 쓰지 않는다.
  (`inverse_gauss_jordan` 이 던지는 `np.linalg.LinAlgError` 예외 타입만 예외)
- 각 함수의 docstring 에 적힌 계약(입력/출력/예외)을 그대로 지킨다.
  노트북의 검증 셀과 `tests/` 가 이 계약을 기준으로 채점된다.
- 구현을 마치면 `raise NotImplementedError(...)` 줄을 지운다.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "as_vector",
    "dot",
    "norm",
    "angle_between",
    "normalize",
    "project",
    "reject",
    "skew",
    "cross",
    "plane_normal",
    "row_echelon",
    "rank",
    "det",
    "gauss_eliminate",
    "inverse_gauss_jordan",
]


# ---------------------------------------------------------------- 기본 연산

def as_vector(v) -> np.ndarray:
    """입력(리스트/튜플/배열)을 1차원 float 배열로 변환한다.

    1차원이 아니면 ValueError 를 던진다.

    [구현 예시] 아래 세 줄이 이 파일에서 기대하는 코드 스타일이다.
    나머지 함수도 이런 식으로 채워 넣으면 된다.
    """
    arr = np.asarray(v, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"1차원 벡터가 필요합니다. 받은 shape={arr.shape}")
    return arr


def dot(a, b) -> float:
    """내적. sum(a_i * b_i) 를 직접 계산한다 (`np.dot` 사용 금지).

    두 벡터의 차원이 다르면 ValueError.
    """
    # TODO: 문제 1-1
    a = as_vector(a)
    b = as_vector(b)

    result = np.sum(a*b)
    return result
    

def norm(v) -> float:
    """유클리드 노름. sqrt(v·v) — 위에서 만든 dot 을 재사용한다."""
    # TODO: 문제 1-1
    v = as_vector(v)

    result = np.sqrt(dot(v,v))

    return result
    

def angle_between(a, b, degrees: bool = True) -> float:
    """두 벡터 사이각. degrees=True 면 도(°), False 면 라디안.

    cos(theta) = (a·b) / (|a||b|)

    주의 1. 영벡터가 들어오면 사이각이 정의되지 않는다 -> ValueError.
    주의 2. 부동소수점 오차로 |cos| 가 1 을 아주 조금 넘으면 arccos 가 nan 을 낸다.
            [-1, 1] 로 clip 해야 무작위 입력에서도 안전하다.
    """
    # TODO: 문제 1-1
    a = as_vector(a)
    b = as_vector(b)
    under = norm(a)*norm(b)
    if under <= 1e-12:
        raise ValueError("영벡터와의 사이각은 정의할 수 없음")
        
    cos_angle = dot(a,b)/under
    cos_angle = np.clip(cos_angle,-1.0,1.0)

    result = np.arccos(cos_angle)

    if degrees:
        return np.degrees(result)
    return result


def normalize(v, eps: float = 1e-12) -> np.ndarray:
    """단위벡터로 정규화한다. v / |v|

    영벡터를 어떻게 처리할지는 **문제 1-2 에서 직접 정한다.**
    노트북 1-2 에서 (1) 아무 처리 없이 나눴을 때 무슨 일이 나는지 관찰하고,
    (2) 선택한 처리 방식과 근거를 마크다운에 적은 뒤, 그 방식대로 여기에 구현한다.
    선택에 따라 노트북/테스트의 검증 코드도 그 방식에 맞춰 작성한다.
    """
    # TODO: 문제 1-2
    v = as_vector(v)
    length = norm(v)

    if length <= eps:
        raise ValueError("영벡터는 정규화할 수 없습니다")
    return v/length


def project(a, b) -> np.ndarray:
    """a 를 b 방향으로 정사영한 성분.

        proj_b(a) = (a·b / b·b) * b

    분모가 |b|^2 이므로 b 를 미리 정규화할 필요는 없다.
    b 가 영벡터면 ValueError.
    """
    # TODO: 문제 1-3
    a = as_vector(a)
    b = as_vector(b)

    if dot(b,b) <= 1e-12:
        raise ValueError("영벡터는 정사영할 수 없습니다.")
    proj = (dot(a,b)/dot(b,b)) * b

    return proj


def reject(a, b) -> np.ndarray:
    """a 에서 b 방향 성분을 뺀 나머지(수직 성분). a = project + reject 가 성립해야 한다."""
    # TODO: 문제 1-3
    a = as_vector(a)
    b = as_vector(b)
    
    return a - project(a,b)


def skew(M) -> np.ndarray:
    """3차원 벡터 a 에 대응하는 반대칭행렬 [a]_x 를 만든다.

        [a]_x = [[  0, -a3,  a2],
                 [ a3,   0, -a1],
                 [-a2,  a1,   0]]

    만족해야 하는 성질: [a]_x @ b == a x b,  [a]_x.T == -[a]_x
    3차원이 아니면 ValueError.
    """
    # TODO: 문제 1-4
    a1,a2,a3 = M[0],M[1],M[2]

    result = np.array([
        [0,-a3,a2],
        [a3,0,-a1],
        [-a2,a1,0]
    ])

    return result


def cross(v1, v2) -> np.ndarray:
    """외적을 **반대칭행렬 곱으로** 계산한다 (`np.cross` 사용 금지)."""
    # TODO: 문제 1-4
    v1 = as_vector(v1)
    v2 = as_vector(v2)
    if v1.shape != (3,) or v2.shape !=(3,):
        raise ValueError("3차원 벡터가 아닙니다")
    else:
        x = v1[1]*v2[2] - v1[2]*v2[1]
        y = v1[2]*v2[0] - v1[0]*v2[2]
        z = v1[0]*v2[1] - v1[1]*v2[0]

    return np.array([x,y,z])


def plane_normal(P1, P2, P3) -> np.ndarray:
    """세 점이 이루는 평면의 **단위** 법선 벡터.

    두 모서리 벡터(P2-P1, P3-P1)의 외적이 평면에 수직이다.
    세 점이 일직선이면 외적이 영벡터가 되어 평면이 하나로 정해지지 않는다 -> ValueError.
    """
    # TODO: 문제 1-5
    P1 = as_vector(P1)
    P2 = as_vector(P2)
    p3 = as_vector(P3)
    
    u = P2-P1
    v = P3-P1

    if norm(cross(u,v)) <= 1e-12:
        raise ValueError("세 점이 일직선입니다")

    normal_vec = normalize(cross(u,v))

    return normal_vec

# ------------------------------------------------- 가우스 소거 기반 선형대수

def row_echelon(M, pivoting: bool = True):
    """행 사다리꼴(row echelon form) 로 만든다.

    Parameters
    ----------
    pivoting : True 면 부분 피벗팅(각 열에서 절댓값이 가장 큰 행을 피벗으로 올림)

    Returns
    -------
    U : (m, n) 상삼각 형태 행렬
    pivot_cols : 피벗이 선 열 인덱스 리스트
    n_swaps : 행 교환 횟수 (행렬식 부호 계산에 필요)

    힌트: 0 인지 판정할 때는 `== 0` 대신 허용오차(tol)를 쓴다.
          예) tol = max(m, n) * np.finfo(float).eps * max(1.0, np.max(np.abs(U)))
    """
    # TODO: 문제 1-6 / 문제 4
    M = np.array(M, dtype=float, copy=True)
    row,col = M.shape
    swap=0
    pivots = []

    r = 0
    c = 0

    while r<row and c < col:
        pivot_idx = r
        while pivot_idx < row and M[pivot_idx,c] == 0.0:
            pivot_idx+=1

        if pivot_idx == row:
            c+=1
            continue

        if pivot_idx != r:
            temp = M[pivot_idx].copy()
            M[pivot_idx] = M[r]
            M[r] = temp
            swap+=1

        pivot = M[r,c]
        pivots.append(pivot)

        for i in range(r+1,row):
            if M[i,c] > 1e-12:
                M[i] = M[r] * (-(M[i,c]/pivot)) + M[i]

        r+=1
        c+=1

    return M,pivots,swap


def rank(A) -> int:
    """행 사다리꼴의 피벗 개수 = rank."""
    # TODO: 문제 1-6
    _,pivots,_ = row_echelon(A)

    return len(pivots)

def det(M) -> float:
    """행렬식 = 행 사다리꼴 대각성분의 곱 x (-1)^(행 교환 횟수).

    피벗이 n 개보다 적으면(특이행렬) 0.0 을 돌려준다.
    정사각 행렬이 아니면 ValueError.
    """
    # TODO: 문제 1-6
    if M.shape == (2,2):
        result = ((M[0,0] * M[1,1]) - (M[0,1] * M[1,0]))

    if M.shape == (3,3):
        result = ((M[0,0] * M[1,1] * M[2,2]) + (M[0,1] * M[1,2] * M[2,0]) + (M[0,2] * M[1,0] * M[2,1])) - ((M[0,2] * M[1,1]* M[2,0]) + (M[0,0] * M[1,2] * M[2,1]) + (M[0,1] * M[1,0] * M[2,2]))

    return result


def gauss_eliminate(A, b, pivoting: bool = True, verbose: bool = False):
    """가우스 소거법 + 후진대입으로 Ax = b 를 푼다.

    Parameters
    ----------
    pivoting : True 면 부분 피벗팅을 적용한다. False 면 피벗을 그대로 쓴다
               (문제 4-4 에서 두 경우의 오차를 비교하므로 **둘 다 동작해야 한다**).
    verbose  : True 면 각 소거 단계의 첨가행렬 [A|b] 를 출력한다
               (문제 4-1 이 요구하는 '단계별 출력').

    Returns
    -------
    x : 해 벡터
    steps : 단계별 첨가행렬 [A|b] 스냅샷 리스트 (초기 상태 포함)

    피벗이 0 이면 해가 유일하지 않다 -> ZeroDivisionError.
    """
    # TODO: 문제 4-1
    A = np.array(A, dtype=float, copy=True)
    b = as_vector(b)
    row,col = A.shape
    M = np.hstack([A,b.reshape(-1,1)])

    steps = [M.copy()]

    for r in range(row):
        if pivoting:
            li = []
            for c in range(r,col):
                li.append(np.abs(M[c,r]))
            bigger_row = np.argmax(li) + r

            if bigger_row != r:
                temp = M[bigger_row].copy()
                M[bigger_row] = M[r]
                M[r] = temp
                
        pivot = M[r,r]
        if pivot == 0.0:
            raise ValueError("피벗이 0이라 해가 무수히 많다")
        
        for i in range(r + 1, row):
            if M[i, r] != 0.0:
                factor = M[i, r] / pivot
                M[i] = M[i] - factor * M[r]

        steps.append(M.copy())

    x = np.zeros(row)
    for i in range(row-1,-1,-1):
        x[i] = (M[i,-1] - dot(M[i, i+1:col], x[i+1:])) / M[i,i]
    
    return x,steps

def inverse_gauss_jordan(A) -> np.ndarray:
    """가우스-조던 소거로 역행렬을 구한다. [A|I] -> [I|A^-1].

    정사각이 아니면 ValueError, 특이행렬이면 np.linalg.LinAlgError.
    (`np.linalg.inv` 를 부르지 말고 소거로 직접 구한다)
    """
    # TODO: 문제 4-3
    A = np.array(A,dtype=float,copy=True)
    row,col = A.shape
    M = np.hstack([A,np.eye(row)])

    for r in range(row):
        li = []
        for c in range(r,row):
            li.append(np.abs(M[c,r]))
        bigger_row = np.argmax(li) + r

        if bigger_row != r:
            temp = M[bigger_row].copy()
            M[bigger_row] = M[r]
            M[r] = temp

        pivot = M[r,r]
        
        if pivot == 0.0:
            raise ValueError("피벗이 0이라 해가 무수히 많다.")

        if pivot != 0:
            M[r] = M[r] * (1/pivot)
            
        for i in range(row):
            if i == r:
                continue
            if M[i,r] != 0:
                factor = M[i,r]
                M[i] = M[i] - M[r]*factor

    return M[:, row:]
