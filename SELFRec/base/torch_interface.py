import torch
import numpy as np
class TorchGraphInterface(object):
    def __init__(self):
        pass

    @staticmethod
    def convert_sparse_mat_to_tensor(X):
        # 将稀疏矩阵转换为 COO 格式
        coo = X.tocoo()

        # 优化：将列表转换为 numpy.ndarray
        indices = np.vstack((coo.row, coo.col))

        # 转换为 PyTorch 张量
        i = torch.LongTensor(indices)
        v = torch.from_numpy(coo.data).float()

        # 使用 torch.sparse_coo_tensor 替代 torch.sparse.FloatTensor
        return torch.sparse_coo_tensor(i, v, coo.shape)