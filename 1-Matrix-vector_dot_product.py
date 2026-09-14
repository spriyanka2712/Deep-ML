#python
def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) ->list[int|float]:
    if len(a[0]) != len(b):
        return -1
    result = []
    for i in range(len(a)):
        val = 0
        for j in range(len(b)):
            val += a[i][j] * b[j]
        result.append(val)
    return result

#Numpy
import numpy as np
def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
    a = np.array(a)
    b = np.array(b)
    if a.shape[1] != b.shape[0]:          
        return -1
    return (a @ b).tolist()

#torch
import torch
def matrix_dot_vector(a, b) -> torch.Tensor:
    a_t = torch.as_tensor(a, dtype=torch.float)
    b_t = torch.as_tensor(b, dtype=torch.float)
    if a_t.size(1) != b_t.size(0):
        return torch.tensor(-1)
    return torch.matmul(a_t, b_t)