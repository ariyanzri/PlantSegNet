import torch
import torch.nn as nn


class DBSCAN(nn.Module):
    def __init__(self, eps, minPts, device_name):
        super().__init__()
        self.eps = eps
        self.minPts = minPts
        self.device_name = device_name

    def forward(self, point_features):
        if len(point_features.shape) == 2:
            point_features = point_features.unsqueeze(0)
        elif len(point_features.shape) != 3:
            raise Exception(
                "Invalid shape for the point_features tensor. It should be either (batch_size, points, features) or (points, features). "
            )

        batch_size = point_features.shape[0]
        point_no = point_features.shape[1]
        dim_size = point_features.shape[2]

        final_clusters = torch.full((batch_size, point_no), 0)

        distance_points = torch.cdist(point_features, point_features)
        neighbors = torch.where(distance_points < self.eps, 1, 0)

        final_clusters[torch.count_nonzero(neighbors, 2) < self.minPts] = -1
