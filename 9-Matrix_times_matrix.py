#python
def matrixmul(a: list[list[int|float]],
              b: list[list[int|float]]) -> list[list[int|float]]:
    if len(a[0]) != len(b):
        return -1
    rows = len(a)
    cols = len(b[0])
    c = [[0] * cols for _ in range(rows)]
    for i in range(len(a)):        
        for j in range(len(b[0])): 
            for k in range(len(b)): 
                c[i][j] += a[i][k] * b[k][j]
    return c

#numpy
import numpy as np
def matrixmul(a: list[list[int|float]],
              b: list[list[int|float]]) -> list[list[int|float]]:
    if np.shape(a)[1] != np.shape(b)[0]:
        return -1
    c = np.dot(a, b)
    return c

#pytorch
import torch
def matrixmul(a, b) -> torch.Tensor:
    if a.size(1) != b.size(0):
        return -1
    return a.matmul(b)