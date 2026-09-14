#python
def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	means = []
	if mode == 'row':
		for i in range(len(matrix)):
			sum = 0
			for j in range(len(matrix[0])):
				sum += matrix[i][j]
			avg = sum / len(matrix[0])
			means.append(avg)

	elif mode == 'column':
		for j in range(len(matrix[0])):
			sum = 0
			for i in range(len(matrix)):
				sum += matrix[i][j]
			avg = sum / len(matrix)
			means.append(avg)	
	return means

#numpy
import numpy as np
def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	if(mode == 'row'):
		means = np.mean(matrix, axis=1)
	else:
		means = np.mean(matrix, axis=0)	
	return means

#pytorch
import torch
def calculate_matrix_mean(matrix, mode: str) -> torch.Tensor:
    a_t = torch.as_tensor(matrix, dtype=torch.float)
    if(mode == 'row'):
        return a_t.mean(dim=1)
    else:
        return a_t.mean(dim=0)
    pass