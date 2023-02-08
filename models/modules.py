import torch
import torch.nn as nn
import torch.nn.functional as F


class KNNSpaceRegularizer(nn.Module):
    def __init__(self, max_k):
        super(KNNSpaceRegularizer, self).__init__()
        self.k_vector = torch.nn.Parameter(torch.randn(max_k))
        self.k_vector.requires_grad = True

    def forward(self, x, preds):
        k = torch.argmax(F.softmax(self.k_vector, 0), dim=0).item() + 1

        batch_size = preds.size(0)
        num_points = preds.size(1)
        num_dims = preds.size(2)

        distance = torch.cdist(x, x)
        idx = torch.topk(distance, k, -1, False)[1]
        idx_base = (
            torch.arange(0, batch_size, device=idx.device).view(-1, 1, 1) * num_points
        )
        idx = idx + idx_base
        idx = idx.view(-1)

        preds = preds.contiguous()
        preds = preds.view(batch_size * num_points, -1)

        knn_coords = preds[idx, :]
        knn_coords = knn_coords.view(batch_size, num_points, k, num_dims)
        new_space = torch.mean(knn_coords, 2)

        return new_space


# import matplotlib.pyplot as plt

# knn_module = KNNSpaceMean(2).to(torch.device("cuda"))

# # points = torch.rand((4, 8000, 3)).to(torch.device("cuda"))
# # preds = torch.rand((4, 8000, 256)).to(torch.device("cuda"))

# points = torch.rand((1, 10, 3)).to(torch.device("cuda"))
# preds = torch.rand((1, 10, 3)).to(torch.device("cuda"))

# new_space = knn_module(points, preds.clone())

# points = points.cpu().numpy()
# preds = preds.cpu().numpy()
# new_space = new_space.cpu().numpy()

# k = torch.argmax(F.softmax(knn_module.k_vector, 0), dim=0).item() + 1
# print(k)
# fig = plt.figure(figsize=(20, 20))
# ax1 = fig.add_subplot(1, 2, 1, projection="3d")
# ax2 = fig.add_subplot(1, 2, 2, projection="3d")
# ax1.scatter(points[:, :, 0], points[:, :, 1], points[:, :, 2], c=preds.squeeze())
# ax2.scatter(points[:, :, 0], points[:, :, 1], points[:, :, 2], c=new_space.squeeze())

# for i, (x, y, z) in enumerate(points.squeeze()):
#     l1 = f"{i}: {preds[0, i, 0]:3.2f}, {preds[0, i, 1]:3.2f}, {preds[0, i, 2]:3.2f}"
#     l2 = f"{new_space[0, i, 0]:3.2f}, {new_space[0, i, 1]:3.2f}, {new_space[0, i, 2]:3.2f}"
#     ax1.text(x, y, z + 0.05, l1, zdir=None)
#     ax2.text(x, y, z + 0.05, l2, zdir=None)

# plt.show()
