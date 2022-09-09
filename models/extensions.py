from turtle import distance
from torch.autograd import Function
import numpy as np
import torch


def multi_random_choice(samples, s_size, max):

    out = np.zeros((samples, s_size))

    for i in range(samples):
        out[i, :] = np.random.choice(max, s_size, replace=True)

    return out


class Cluster(Function):
    @staticmethod
    def forward(ctx, input, dist):
        """
        Input : BXNX256
        Output: BXN
        """

        distance_pred = torch.cdist(input, input)
        distance_pred = torch.where(distance_pred < dist, 1, 0)
        pred_final_cluster = torch.zeros(
            (distance_pred.shape[0], distance_pred.shape[1]), dtype=torch.int16
        )

        next_label = 1
        for b in range(pred_final_cluster.shape[0]):
            for i in range(pred_final_cluster.shape[1]):
                if pred_final_cluster[b, i] == 0:
                    pred_final_cluster[b, i] = next_label
                    next_label += 1
                value = pred_final_cluster[b, i].clone()
                pred_final_cluster[b, distance_pred[b, i] == 1] = value
        return pred_final_cluster

    @staticmethod
    def backward(ctx, grad_output):
        """
        Input : BXN
        Output: BXNX256
        """
        grad_input = grad_output.unsqueeze(-1).repeat(1, 1, 256)
        return grad_input, None
        # input (B, N, 1)
        # Todo do we duplicate the grad value or split?
        # return (B, N, 256)


class Downsample(Function):
    @staticmethod
    def forward(ctx, input, result_size):
        ds_dim_size = input.shape[1]
        batch_size = input.shape[0]
        ds_idx = multi_random_choice(batch_size, result_size, ds_dim_size)
        ds_idx = np.expand_dims(ds_idx, axis=2)
        gather_idx = torch.tensor(
            np.repeat(ds_idx, 3, axis=2), dtype=torch.int64
        ).cuda()
        ctx.save_for_backward(input, gather_idx)
        output = torch.gather(input, 1, gather_idx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, gather_idx = ctx.saved_tensors
        # grad_input = torch.zeros(input.shape)
        # grad_input = grad_input.scatter(1, gather_idx, grad_output)
        # return grad_input, None
        return torch.full(input.shape, torch.flatten(grad_output)[0]), None


class LeafBranchFunc(Function):
    @staticmethod
    def forward(ctx, point_features, points, dist, leaf_classifier):

        cluster_assignments = Cluster.apply(point_features, dist)
        gradient_batch_points = torch.zeros(
            point_features.shape, requires_grad=True
        ).cuda()
        losses = torch.zeros(cluster_assignments.shape[0])
        for batch in range(cluster_assignments.shape[0]):
            clusters = cluster_assignments[batch].unique()
            classifications = []
            for cluster in clusters:
                cluster_points = points[
                    batch, cluster_assignments[batch] == cluster
                ].unsqueeze(0)
                cluster_points_ds = Downsample.apply(cluster_points, 80)
                leaf_cls = torch.nn.functional.softmax(
                    leaf_classifier(cluster_points_ds)
                )
                classifications.append(leaf_cls[0, 1])
                gradient_batch_points[cluster_assignments == cluster, :] = torch.log(
                    leaf_cls[0, 1]
                )
                # gradient_batch_points[cluster_assignments == cluster, :] = leaf_cls[
                #     0, 1
                # ]
            classifications = torch.nan_to_num(torch.tensor(classifications), 0)
            expected_labels = torch.ones(classifications.shape)
            loss = torch.nn.functional.binary_cross_entropy(
                classifications, expected_labels
            )
            # loss = expected_labels - classifications
            losses[batch] = loss
        ctx.save_for_backward(gradient_batch_points)
        return losses.sum()

    @staticmethod
    def backward(ctx, grad_out):
        # input size BX1
        # output size BXNX256
        gradient_batch_points = ctx.saved_tensors[0]
        gradient_batch_points = torch.zeros(gradient_batch_points.shape).cuda()
        return gradient_batch_points, None, None, None


class LeafBranch(torch.nn.Module):
    def __init__(self, dist, leaf_classifier) -> None:
        super().__init__()
        self.dist = dist
        self.leaf_classifier_model = leaf_classifier
        self.fn = LeafBranchFunc.apply

    def forward(self, point_features, points):
        return self.fn(point_features, points, self.dist, self.leaf_classifier_model)
