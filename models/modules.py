import torch
import torch.nn as nn
import torch.nn.functional as F


class KNNSpaceMean(nn.Module):
    def __init__(self):
        super(KNNSpaceMean, self).__init__()
        self.k_vector = torch.nn.Parameter(torch.randn(2))
        self.k_vector.requires_grad = True

    def forward(self, points, preds):
        k = torch.argmax(F.softmax(self.k_vector, 0), dim=0) + 1
        k = 2
        distance = torch.cdist(points, points)
        k_nearest_ind = torch.topk(distance, k, 2, False)[1]
        k_nearest_ind = k_nearest_ind.unsqueeze(-1).expand(
            *k_nearest_ind.shape, preds.shape[2]
        )
        preds = preds.unsqueeze(1).expand(
            (preds.shape[0], preds.shape[1], preds.shape[1], preds.shape[2])
        )
        knn_preds = preds.gather(2, k_nearest_ind)
        new_space = torch.mean(knn_preds, 2)
        return new_space


import matplotlib.pyplot as plt

knn_module = KNNSpaceMean()

points = torch.rand((4, 8000, 3))
preds = torch.rand((4, 8000, 256))

# points = torch.rand((1, 10, 3))
# preds = torch.rand((1, 10, 3))

new_space = knn_module(points, preds)

# fig = plt.figure(figsize=(20, 20))
# ax1 = fig.add_subplot(1, 2, 1, projection="3d")
# ax2 = fig.add_subplot(1, 2, 2, projection="3d")
# ax1.scatter(points[:, :, 0], points[:, :, 1], points[:, :, 2], c=preds.squeeze())
# ax2.scatter(points[:, :, 0], points[:, :, 1], points[:, :, 2], c=new_space.squeeze())
# plt.show()
