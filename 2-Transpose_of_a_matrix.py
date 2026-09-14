#python
def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    transposed_a = [[0 for i in range(len(a))] for j in range(len(a[0]))]
    for i in range(len(a)):
        for j in range(len(a[i])):
            transposed_a[j][i] = a[i][j]
    return transposed_a
    pass

#numpy
import numpy as np
def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    a_np = np.array(a)
    return np.transpose(a_np)

#pytorch
import torch
def transpose_matrix(a) -> torch.Tensor:
    a_t = torch.as_tensor(a)
    return a_t.T