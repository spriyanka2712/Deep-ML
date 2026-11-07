#numpy
import numpy as np
def pca(data: np.ndarray, k: int) -> np.ndarray:
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std_data = (data - mean) / std
    cov_matrix = np.cov(std_data, rowvar=False)
    evals, evecs = np.linalg.eigh(cov_matrix)
    sorted_indices = np.argsort(evals)[::-1]
    sorted_evecs = evecs[:, sorted_indices]
    top_k = sorted_evecs[:, :k]
    for i in range(top_k.shape[1]):
        if top_k[0, i] < 0:
            top_k[:, i] *= -1
    return np.round(top_k, 4)

#pytorch
import torch
def pca(data, k) -> torch.Tensor:
    data_t = torch.as_tensor(data, dtype=torch.float)
    mean = data_t.mean(dim=0, keepdim=True)
    std  = data_t.std(dim=0, 
    unbiased=False, keepdim=True)
    data_std = (data_t - mean) / std
    cov = torch.cov(data_std.T)
    eigenvalues, eigenvectors = torch.linalg.eig(cov)
    vals = eigenvalues.real
    vecs = eigenvectors.real
    idx = torch.argsort(vals, descending=True)
    pcs = vecs[:, idx[:k]]
    pcs = torch.round(pcs * 10000) / 10000
    return pcs