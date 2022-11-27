import torch
import torch.nn as nn
from torchsummary import summary
from collections import namedtuple
import numpy as np
import torch.nn.functional as F


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False, device_name="cuda"):
    batch_size = x.size(0)
    num_points = x.size(2)

    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device(device_name)

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(
        2, 1
    ).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature  # (batch_size, 2*num_dims, num_points, k)


class DGCNNBinaryClass(nn.Module):
    def __init__(self, args):
        print(args)
        super(DGCNNBinaryClass, self).__init__()
        self.args = args
        self.k = args.k
        self.num_points = args.num_points

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(self.num_points * 256, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 2)

    def forward(self, x):

        x = x.transpose(1, 2)

        batch_size = x.size(0)
        num_points = x.size(2)

        x = get_graph_feature(
            x, k=self.k
        )  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(
            x
        )  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[
            0
        ]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(
            x1, k=self.k
        )  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(
            x
        )  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[
            0
        ]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(
            x2, k=self.k
        )  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(
            x
        )  # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[
            0
        ]  # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(
            x3, k=self.k
        )  # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(
            x
        )  # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[
            0
        ]  # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = self.flatten(x4)

        x5 = F.relu(self.fc1(x))
        x6 = F.relu(self.fc2(x5))
        x7 = self.fc3(x6)

        return x7


class DGCNNFeatureSpace(nn.Module):
    def __init__(self, args, input_dim=3, device_name="cuda"):
        super(DGCNNFeatureSpace, self).__init__()
        self.args = args
        self.device = device_name
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim * 2, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x):
        x = x.transpose(1, 2)

        x = get_graph_feature(
            x, k=self.k, device_name=self.device
        )  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(
            x
        )  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[
            0
        ]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(
            x1, k=self.k, device_name=self.device
        )  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(
            x
        )  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[
            0
        ]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(
            x2, k=self.k, device_name=self.device
        )  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(
            x
        )  # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[
            0
        ]  # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(
            x3, k=self.k, device_name=self.device
        )  # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(
            x
        )  # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[
            0
        ]  # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        return x4.transpose(1, 2)


class DGCNNSemanticSegmentor(nn.Module):
    def __init__(self, k, output_dim=3, input_dim=3, device_name="cuda"):
        super(DGCNNSemanticSegmentor, self).__init__()
        self.device = device_name
        self.k = k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(output_dim)

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim * 2, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, output_dim, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x):
        x = x.transpose(1, 2)

        x = get_graph_feature(
            x, k=self.k, device_name=self.device
        )  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(
            x
        )  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[
            0
        ]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(
            x1, k=self.k, device_name=self.device
        )  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(
            x
        )  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[
            0
        ]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(
            x2, k=self.k, device_name=self.device
        )  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(
            x
        )  # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[
            0
        ]  # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(
            x3, k=self.k, device_name=self.device
        )  # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(
            x
        )  # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[
            0
        ]  # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        return x4


# MyStruct = namedtuple('args', 'k')
# args = MyStruct(k=15)
# DGCNN = DGCNNSemanticSegmentor(args).cuda()

# input = torch.Tensor(np.random.rand(2,8000,3)).cuda()
# out = DGCNN(input)
# print(out.shape)
