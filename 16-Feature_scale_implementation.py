#numpy
import numpy as np
def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
	mean = np.mean(data, axis = 0)
	stdev = np.std(data, axis = 0)
	standardized_data = (data - mean) / stdev
	minimum = np.min(data, axis = 0)
	maximum = np.max(data, axis = 0)
	normalized_data = (data - minimum) / (maximum - minimum)
	return np.round(standardized_data, 4), np.round(normalized_data, 4)

#pytorch
import torch
def feature_scaling(data) -> tuple[torch.Tensor, torch.Tensor]:
    data_t = torch.as_tensor(data, dtype=torch.float)
    mean = data_t.mean(dim = 0)
    std = data_t.std(dim = 0, unbiased = False)
    standardized_data = (data_t - mean) / std
    minimum = data_t.min(dim = 0).values
    maximum = data_t.max(dim = 0).values
    normalized_data = (data_t - minimum) / (maximum - minimum)
    return torch.round(standardized_data*10000)/10000, torch.round(normalized_data*10000)/10000