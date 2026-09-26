#numpy
import numpy as np
def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:
	t_inv = np.linalg.inv(T)
	if np.linalg.det(T) == 0 or np.linalg.det(S) == 0:
		return -1
	transformed_matrix = t_inv @ A @ S
	return transformed_matrix

#pytorch
import torch
def transform_matrix(A, T, S) -> torch.Tensor:
    A_t = torch.as_tensor(A, dtype=torch.float)
    T_t = torch.as_tensor(T, dtype=torch.float)
    S_t = torch.as_tensor(S, dtype=torch.float)
    if torch.linalg.det(T_t) != 0 and torch.linalg.det(S_t) != 0:
        x = torch.matmul(torch.linalg.inv(T_t),A_t)
        y = torch.matmul(x,S_t)
        return y
    else:
        return torch.tensor(-1.)