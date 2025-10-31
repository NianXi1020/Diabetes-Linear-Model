"""Lightweight linear algebra utilities using pure Python."""
from __future__ import annotations

import math
from typing import Iterable, List

Vector = List[float]
Matrix = List[List[float]]


def zeros(rows: int, cols: int) -> Matrix:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def identity(n: int) -> Matrix:
    mat = zeros(n, n)
    for i in range(n):
        mat[i][i] = 1.0
    return mat


def transpose(matrix: Matrix) -> Matrix:
    if not matrix:
        return []
    return [list(row) for row in zip(*matrix)]


def matmul(A: Matrix, B: Matrix) -> Matrix:
    result = zeros(len(A), len(B[0]))
    for i, row in enumerate(A):
        for k, a in enumerate(row):
            if a == 0.0:
                continue
            for j, b in enumerate(B[k]):
                result[i][j] += a * b
    return result


def matvec(A: Matrix, v: Vector) -> Vector:
    return [sum(a * b for a, b in zip(row, v)) for row in A]


def vecdot(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b))


def cholesky(A: Matrix) -> Matrix:
    n = len(A)
    L = zeros(n, n)
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                value = A[i][i] - s
                if value <= 0:
                    raise ValueError("Matrix is not positive definite.")
                L[i][j] = math.sqrt(value)
            else:
                if L[j][j] == 0:
                    raise ValueError("Matrix is singular.")
                L[i][j] = (A[i][j] - s) / L[j][j]
    return L


def solve_lower_triangular(L: Matrix, b: Vector) -> Vector:
    y = [0.0] * len(L)
    for i in range(len(L)):
        s = sum(L[i][k] * y[k] for k in range(i))
        y[i] = (b[i] - s) / L[i][i]
    return y


def solve_upper_triangular(U: Matrix, b: Vector) -> Vector:
    n = len(U)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(U[i][k] * x[k] for k in range(i + 1, n))
        x[i] = (b[i] - s) / U[i][i]
    return x


def solve_pos_def(A: Matrix, b: Vector) -> Vector:
    L = cholesky(A)
    y = solve_lower_triangular(L, b)
    x = solve_upper_triangular(transpose(L), y)
    return x


def invert_pos_def(A: Matrix) -> Matrix:
    n = len(A)
    inv = zeros(n, n)
    L = cholesky(A)
    Lt = transpose(L)
    for i in range(n):
        e = [0.0] * n
        e[i] = 1.0
        y = solve_lower_triangular(L, e)
        x = solve_upper_triangular(Lt, y)
        for j in range(n):
            inv[j][i] = x[j]
    return inv


def scale_vector(v: Vector, scalar: float) -> Vector:
    return [scalar * x for x in v]


def add_vectors(a: Vector, b: Vector) -> Vector:
    return [x + y for x, y in zip(a, b)]


def quantiles(values: List[Vector], probs: Iterable[float]) -> List[Vector]:
    if not values:
        return []
    sorted_values = [sorted(col) for col in zip(*values)]
    n = len(values)
    result: List[Vector] = []
    for p in probs:
        if not 0.0 <= p <= 1.0:
            raise ValueError("Probabilities must lie in [0, 1].")
        index = p * (n - 1)
        lower = int(math.floor(index))
        upper = int(math.ceil(index))
        weight = index - lower
        quantile = [
            (1 - weight) * sorted_values[j][lower] + weight * sorted_values[j][upper]
            for j in range(len(sorted_values))
        ]
        result.append(quantile)
    return result


def vector_quantiles(values: List[float], probs: Iterable[float]) -> List[float]:
    if not values:
        return []
    sorted_vals = sorted(values)
    n = len(values)
    result = []
    for p in probs:
        if not 0.0 <= p <= 1.0:
            raise ValueError("Probabilities must lie in [0, 1].")
        index = p * (n - 1)
        lower = int(math.floor(index))
        upper = int(math.ceil(index))
        weight = index - lower
        result.append((1 - weight) * sorted_vals[lower] + weight * sorted_vals[upper])
    return result


def vector_mean(values: List[Vector]) -> Vector:
    if not values:
        return []
    length = len(values[0])
    totals = [0.0] * length
    for value in values:
        for i, x in enumerate(value):
            totals[i] += x
    return [total / len(values) for total in totals]


def vector_variance(values: List[float]) -> float:
    mean = sum(values) / len(values)
    return sum((x - mean) ** 2 for x in values) / len(values)
