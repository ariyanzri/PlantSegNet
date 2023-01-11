import torch.nn as nn
import torch.optim.lr_scheduler as lr_sched
import torch
import numpy as np
from torchmetrics import Accuracy
from torchmetrics.functional import f1, precision_recall
from sklearn.cluster import DBSCAN


def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn


class BNMomentumScheduler(lr_sched.LambdaLR):
    def __init__(self, model, bn_lambda, last_epoch=-1, setter=set_bn_momentum_default):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(type(model)._name_)
            )

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))

    def state_dict(self):
        return dict(last_epoch=self.last_epoch)

    def load_state_dict(self, state):
        self.last_epoch = state["last_epoch"]
        self.step(self.last_epoch)


class SpaceSimilarityLossV2(nn.Module):
    def __init__(self, M1=1, M2=10):
        super().__init__()
        self.M1 = M1
        self.M2 = M2

    def forward(self, input, target):

        distance_pred = torch.cdist(input, input)

        if target.shape[-1] != 1:
            target = torch.unsqueeze(target.float(), dim=-1)
        else:
            target = target.float()
        distances_gt = torch.cdist(target, target)

        normalized_distance_gt = torch.where(
            distances_gt == 0,
            torch.clamp(distance_pred - self.M1, min=0),
            torch.clamp(self.M2 - distance_pred, min=0),
        )

        return torch.mean(normalized_distance_gt)


# Radius close point loss function
class SpaceSimilarityLossV3(nn.Module):
    def __init__(self, points, M1=1, M2=10):
        super().__init__()
        self.M1 = M1
        self.M2 = M2
        self.euclidean_distances = torch.cdist(points, points)

    def forward(self, input, target):

        distance_pred = torch.cdist(input, input)

        if target.shape[-1] != 1:
            target = torch.unsqueeze(target.float(), dim=-1)
        else:
            target = target.float()
        distances_gt = torch.cdist(target, target)

        normalized_distance_gt = torch.where(
            distances_gt == 0,
            torch.clamp(distance_pred - self.M1, min=0),
            torch.clamp(self.M2 - distance_pred, min=0),
        )

        normal_loss = torch.mean(normalized_distance_gt)
        normalized_euclidean_distance = torch.clamp(
            -torch.log(5 * (self.euclidean_distances + 0.01)), min=0
        )
        close_points_loss = torch.mean(
            torch.mul(distance_pred, normalized_euclidean_distance)
        )

        return normal_loss + close_points_loss


# KNN based close point loss function
class SpaceSimilarityLossV4(nn.Module):
    def __init__(self, points, M1=1, M2=10, cpc=0.2):
        super().__init__()
        self.close_point_coef = cpc
        self.M1 = M1
        self.M2 = M2
        self.euclidean_distances = torch.cdist(points, points)

    def forward(self, input, target):

        distance_pred = torch.cdist(input, input)

        if target.shape[-1] != 1:
            target = torch.unsqueeze(target.float(), dim=-1)
        else:
            target = target.float()
        distances_gt = torch.cdist(target, target)

        normalized_distance_gt = torch.where(
            distances_gt == 0,
            torch.clamp(distance_pred - self.M1, min=0),
            torch.clamp(self.M2 - distance_pred, min=0),
        )

        normal_loss = torch.mean(normalized_distance_gt)

        knn_ind = (self.euclidean_distances * -1).topk(k=10, dim=-1)[1]

        knn_pred_dist = distance_pred.gather(-1, knn_ind)

        close_points_loss = torch.mean(knn_pred_dist)

        return normal_loss + self.close_point_coef * close_points_loss


class LeafMetrics(nn.Module):
    def __init__(self, device_name="cpu"):
        super().__init__()
        self.device_name = device_name

    def forward(self, input, target):

        if len(input.shape) != 3:
            raise Exception(
                f"Incorrect shape of the input tensor. It should have 3 dimensions (BXNX1) but it has shape {input.shape}. "
            )

        if len(target.shape) != 3:
            raise Exception(
                f"Incorrect shape of the target tensor. It should have 3 dimensions (BXNX1) but it has shape {target.shape}. "
            )

        cluster_pred = torch.cdist(input.float(), input.float())
        cluster_gt = torch.cdist(target.float(), target.float())

        ones = torch.ones(cluster_pred.shape).int().to(torch.device(self.device_name))
        zeros = torch.zeros(cluster_pred.shape).int().to(torch.device(self.device_name))

        cluster_gt = (
            torch.where(cluster_gt == 0, ones, zeros)
            .squeeze()
            .flatten()
            .to(torch.device(self.device_name))
        )
        cluster_pred = (
            torch.where(cluster_pred == 0, ones, zeros)
            .squeeze()
            .flatten()
            .to(torch.device(self.device_name))
        )

        acc_func = Accuracy().to(torch.device(self.device_name))

        Acc = acc_func(cluster_pred, cluster_gt)
        Precision, Recall = precision_recall(cluster_pred, cluster_gt, multiclass=False)
        F = f1(cluster_pred, cluster_gt, multiclass=False)

        return Acc.item(), Precision.item(), Recall.item(), F.item()


class LeafMetricsTraining(nn.Module):
    def __init__(self, dist):
        super().__init__()
        self.threshold = dist

    def forward(self, input, target):

        cluster_pred = torch.cdist(input, input)

        if target.shape[-1] != 1:
            target = torch.unsqueeze(target.float(), dim=-1)
        else:
            target = target.float()

        cluster_gt = torch.cdist(target, target)
        ones = torch.ones(cluster_pred.shape).cuda()
        zeros = torch.zeros(cluster_pred.shape).cuda()

        TP = torch.sum(
            torch.where(
                (cluster_gt == 0) & (cluster_pred < self.threshold), ones, zeros
            )
        )
        TN = torch.sum(
            torch.where(
                (cluster_gt > 0) & (cluster_pred >= self.threshold), ones, zeros
            )
        )
        FP = torch.sum(
            torch.where((cluster_gt > 0) & (cluster_pred < self.threshold), ones, zeros)
        )
        FN = torch.sum(
            torch.where(
                (cluster_gt == 0) & (cluster_pred >= self.threshold), ones, zeros
            )
        )

        Acc = (TP + TN) / (TP + FP + TN + FN)
        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)
        F = 2 * (Precision * Recall) / (Precision + Recall)

        return Acc.item(), Precision.item(), Recall.item(), F.item()


class SemanticMetrics(nn.Module):
    def __init__(self):
        super().__init__()
        self.accuracy_fn = Accuracy()

    def forward(self, input, target):

        input = input.int().cpu().squeeze()
        target = target.int().cpu().squeeze()

        Acc = self.accuracy_fn(input, target)

        return Acc.item()


class AveragePrecision(nn.Module):
    def __init__(self, iou_th, device_name="cpu"):
        super().__init__()
        self.device_name = device_name
        self.iou_threshold = iou_th

    def forward(self, input, target):

        if len(input.shape) != 1:
            raise Exception(
                f"Incorrect shape of the input tensor. It should have 1 dimensions (N) but it has shape {input.shape}. "
            )

        if len(target.shape) != 1:
            raise Exception(
                f"Incorrect shape of the target tensor. It should have 1 dimensions (N) but it has shape {target.shape}. "
            )

        gt_cluster_labels = list(set(target.int().cpu().numpy().tolist()))
        pr_cluster_labels = list(set(input.int().cpu().numpy().tolist()))

        average_precision = 0
        for gt_cluster in gt_cluster_labels:
            TP = 0
            FP = 0
            for pr_cluster in pr_cluster_labels:
                pr_point_indices = (
                    (input == pr_cluster).nonzero().squeeze().cpu().numpy()
                )
                gt_point_indices = (
                    (target == gt_cluster).nonzero().squeeze().cpu().numpy()
                )
                intersection = np.intersect1d(pr_point_indices, gt_point_indices, True)
                union = np.union1d(pr_point_indices, gt_point_indices)
                iou = len(intersection) / len(union)
                if iou >= self.iou_threshold:
                    TP += 1
                elif iou > 0:
                    FP += 1
            precision = TP / (TP + FP)
            average_precision += precision

        return average_precision / len(gt_cluster_labels)


def binary_acc(out, target):
    return (torch.softmax(out, dim=1).argmax(dim=1) == target).sum().float() / float(
        target.size(0)
    )
