#python
def inverse_2x2(matrix: list[list[float]]) -> list[list[float]] | None:
    a, b, c, d = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
    det = a*d - b*c 
    if(det == 0):
        return None
    return [[d/det, -b/det], [-c/det, a/det]]

#numpy
import numpy as np
def inverse_2x2(matrix: list[list[float]]) -> list[list[float]] | None:
    if np.linalg.det(matrix) == 0:
        return None
    return np.linalg.inv(matrix)

#pytorch
import torch
def inverse_2x2(matrix) -> torch.Tensor | None:
    m = torch.as_tensor(matrix, dtype=torch.float)
    if torch.det(m) == 0:
        return None
    return torch.linalg.inv(m)