#numpy
import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	X, y = np.array(X), np.array(y)
	theta = np.linalg.inv(X.T @ X) @ X.T @ y
	return theta

#pytorch
import torch
def linear_regression_normal_equation(X, y) -> torch.Tensor:
    X_t = torch.as_tensor(X, dtype=torch.float)
    y_t = torch.as_tensor(y, dtype=torch.float).reshape(-1,1)
    theta = torch.inverse(X_t.T @ X_t) @ X_t.T @ y_t
    return theta