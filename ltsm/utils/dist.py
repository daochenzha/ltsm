import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import torch

def pairwise_dtw(x_batch, y_batch):
    """

    Args:
    :param x_batch: Tensor, [ Batchsize, Time, Dimension_x ]
    :param y_batch: Tensor, [ Batchsize, Time, Dimension_y ]

        The input tensor should have Dimension_x == Dimension_y

    :return: Pair-wise Distance, Tensor, [ Batchsize, Batchsize ]
    """

    batchsize_x = x_batch.shape[0]
    batchsize_y = y_batch.shape[0]
    dist_matrix = torch.zeros((batchsize_x, batchsize_y), device=torch.device("cpu"))
    for idx1, x in enumerate(x_batch):
        for idx2, y in enumerate(y_batch):
            if x_batch is y_batch and dist_matrix[idx2, idx1] > 0:
                dist_matrix[idx1, idx2] = dist_matrix[idx2, idx1]

            else:
                distance_xy, _ = fastdtw(x, y, dist=euclidean)
                dist_matrix[idx1, idx2] = distance_xy





