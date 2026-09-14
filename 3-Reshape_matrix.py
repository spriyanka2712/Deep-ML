#python
def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	if(len(a)*len(a[0]) != new_shape[0]*new_shape[1]):
		return []
	
	flatten = []
	for i in range(len(a)):
		for j in range(len(a[0])):
			flatten.append(a[i][j])

	reshaped_matrix = []
	index = 0
	for i in range(new_shape[0]):
		new_row = []
		for j in range(new_shape[1]):
			new_row.append(flatten[index])
			index += 1
		reshaped_matrix.append(new_row)
	return reshaped_matrix

#numpy
import numpy as np
def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	if(len(a)*len(a[0]) != new_shape[0]*new_shape[1]):
		return []
	reshaped_matrix = np.reshape(a, (new_shape[0], new_shape[1]))
	return reshaped_matrix

#pytorch
import torch
def reshape_matrix(a, new_shape) -> torch.Tensor:
    if len(a) * len(a[0]) != new_shape[0] * new_shape[1]:
        return torch.tensor([])
    a_t = torch.as_tensor(a, dtype=torch.float)
    return a_t.reshape(new_shape)