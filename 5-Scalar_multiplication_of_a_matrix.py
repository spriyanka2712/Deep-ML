#python
def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
	for i in range(len(matrix)):
		for j in range(len(matrix[0])):
			matrix[i][j] = matrix[i][j] * scalar
	return matrix
	
#numpy
import numpy as np
def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
	matrix_np = np.array(matrix)
	result = scalar*matrix_np
	return result

#pytorch
import torch
def scalar_multiply(matrix, scalar) -> torch.Tensor:
    m_t = torch.as_tensor(matrix, dtype=torch.float)
    return scalar * matrix