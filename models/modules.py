import torch
import torch.nn as nn
import torch.nn.functional as F


class KNNSpaceMean(nn.Module):
    def __init__(self, max_k):
        super(KNNSpaceMean, self).__init__()
        self.k_vector = torch.nn.Parameter(torch.randn(max_k))
        self.k_vector.requires_grad = True

    def forward(self, points, preds):
        k = torch.argmax(F.softmax(self.k_vector, 0), dim=0) + 1
        distance = torch.cdist(points, points)
        k_nearest_ind = (
            (torch.topk(distance, k, 2, False)[1])
            .unsqueeze(-1)
            .expand(-1, -1, -1, preds.shape[2])
        )
        new_space = torch.mean(
            (
                preds.unsqueeze(1)
                .expand(
                    (preds.shape[0], preds.shape[1], preds.shape[1], preds.shape[2])
                )
                .gather(2, k_nearest_ind)
            ),
            2,
        )
        return new_space
        # k = torch.argmax(F.softmax(self.k_vector, 0), dim=0) + 1
        # x = torch.cdist(points, points)
        # x = torch.topk(x, k, 2, False)[1]
        # x = x.unsqueeze(-1).expand(*x.shape, preds.shape[2])
        # preds = preds.unsqueeze(1).expand(
        #     (preds.shape[0], preds.shape[1], preds.shape[1], preds.shape[2])
        # )
        # preds = preds.gather(2, x)
        # preds = torch.mean(preds, 2)
        # return preds


import matplotlib.pyplot as plt

knn_module = KNNSpaceMean(10).to(torch.device("cuda"))

# points = torch.rand((4, 8000, 3)).to(torch.device("cuda"))
# preds = torch.rand((4, 8000, 256)).to(torch.device("cuda"))

points = torch.rand((1, 10, 3)).to(torch.device("cuda"))
preds = torch.rand((1, 10, 3)).to(torch.device("cuda"))

new_space = knn_module(points, preds)

points = points.cpu().numpy()
preds = preds.cpu().numpy()
new_space = new_space.cpu().numpy()

fig = plt.figure(figsize=(20, 20))
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax2 = fig.add_subplot(1, 2, 2, projection="3d")
ax1.scatter(points[:, :, 0], points[:, :, 1], points[:, :, 2], c=preds.squeeze())
ax2.scatter(points[:, :, 0], points[:, :, 1], points[:, :, 2], c=new_space.squeeze())
plt.show()
