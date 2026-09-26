#python


#numpy
import numpy as np
def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	return np.cov(vectors)

#pytorch
import torch
def calculate_covariance_matrix(vectors) -> torch.Tensor:
    v_t = torch.as_tensor(vectors, dtype=torch.float)
    return torch.cov(v_t)