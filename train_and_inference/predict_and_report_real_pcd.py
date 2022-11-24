import os
import numpy as np
import torch
import open3d as o3d
import torch.nn.functional as F
import sys
import random

sys.path.append("..")
from models.nn_models import *
from models.datasets import SorghumDataset
from models.utils import LeafMetrics
from data.utils import create_ply_pcd_from_points_with_labels
from data.load_raw_data import load_real_ply_with_labels
from scipy.spatial.distance import cdist
from scipy import stats
import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="Sorghum 3D part segmentation prediction script for real data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-s",
        "--semantic_version",
        help="The version of the semantic segmentation model. If not determined, the latest version would be picked.",
        metavar="semantic_version",
        required=False,
        type=int,
        default=-1,
    )

    parser.add_argument(
        "-i",
        "--instance_version",
        help="The version of the instance segmentation model. If not determined, the latest version would be picked.",
        metavar="instance_version",
        required=False,
        type=int,
        default=-1,
    )

    parser.add_argument(
        "-d",
        "--dist",
        help="Distance threshold below which points are considered to be in the same cluster.",
        metavar="dist",
        required=False,
        type=float,
        default=1,
    )

    parser.add_argument(
        "-p",
        "--path",
        help="Path to the dataset folder.",
        metavar="path",
        required=False,
        type=str,
        default="/space/ariyanzarei/sorghum_segmentation/dataset/real_data/labeled/ply_files",
    )

    parser.add_argument(
        "-v",
        "--save",
        help="Whether to save the predicted point clouds or not",
        action="store_true",
    )

    return parser.parse_args()


def predict_downsampled(points, semantic_model, instance_model, dist=5):
    if (
        "use_normals" in semantic_model.hparams
        and semantic_model.hparams["use_normals"]
    ):
        pred_semantic_label = semantic_model(torch.unsqueeze(points, dim=0).cuda())
    else:
        pred_semantic_label = semantic_model(
            torch.unsqueeze(points[:, :3], dim=0).cuda()
        )

    pred_semantic_label = F.softmax(pred_semantic_label, dim=1)
    pred_semantic_label = pred_semantic_label[0].cpu().detach().numpy().T
    pred_semantic_label_labels = np.argmax(pred_semantic_label, 1)

    instance_points = points[pred_semantic_label_labels == 1, :]

    if (
        "use_normals" in instance_model.hparams
        and instance_model.hparams["use_normals"]
    ):
        pred_instance_features = instance_model(
            torch.unsqueeze(instance_points, dim=0).cuda()
        )
    else:
        pred_instance_features = instance_model(
            torch.unsqueeze(instance_points[:, :3], dim=0).cuda()
        )

    distance_pred = torch.cdist(pred_instance_features, pred_instance_features)
    distance_pred = distance_pred.cpu().detach().numpy().T
    distance_pred = np.squeeze(distance_pred)

    distance_pred = 1 * (distance_pred < dist)

    pred_final_cluster = np.zeros((distance_pred.shape[0])).astype("uint16")
    next_label = 1

    for i in range(distance_pred.shape[0]):
        if pred_final_cluster[i] == 0:
            pred_final_cluster[i] = next_label
            next_label += 1

        ind = np.where(distance_pred[i] == 1)

        for j in ind[0]:
            pred_final_cluster[j] = pred_final_cluster[i]

    print("Number of predicted leaf instances: ", len(list(set(pred_final_cluster))))

    points = points.cpu().detach().numpy()
    instance_points = instance_points.cpu().detach().numpy()

    ply_semantic = create_ply_pcd_from_points_with_labels(
        points[:, :3], pred_semantic_label_labels, is_semantic=True
    )

    ply_instance = create_ply_pcd_from_points_with_labels(
        instance_points[:, :3], pred_final_cluster
    )

    return ply_semantic, ply_instance, pred_semantic_label_labels, pred_final_cluster


def pred_full_size(
    full_points, downsampled_points, downsampled_semantic, downsampled_instance, k=10
):

    distances = cdist(full_points, downsampled_points)
    full_ind, down_ind = np.where(distances == 0)

    semantic_full = np.ones((full_points.shape[0])).astype("uint8") * -1
    semantic_full[full_ind] = downsampled_semantic[down_ind]

    for i in range(full_points.shape[0]):
        sorted_distances = np.argsort(distances[i])
        if semantic_full[i] == -1:
            mode = stats.mode(downsampled_semantic[sorted_distances[:k]])[0]
            semantic_full[i] = mode

    semantic_ply = create_ply_pcd_from_points_with_labels(
        full_points, semantic_full, is_semantic=True
    )

    downsampled_focal_points = downsampled_points[downsampled_semantic == 1]
    focal_points = full_points[semantic_full == 1]

    distances = cdist(focal_points, downsampled_focal_points)
    full_ind, down_ind = np.where(distances == 0)

    instance_full = np.ones((focal_points.shape[0])).astype("uint8") * -1
    instance_full[full_ind] = downsampled_instance[down_ind]

    for i in range(focal_points.shape[0]):
        sorted_distances = np.argsort(distances[i])
        if instance_full[i] == -1:
            mode = stats.mode(downsampled_instance[sorted_distances[:k]])[0]
            instance_full[i] = mode

    instance_ply = create_ply_pcd_from_points_with_labels(focal_points, instance_full)

    return semantic_ply, instance_ply


def save_predicted(ply_pcd, path):
    o3d.io.write_point_cloud(path, ply_pcd)


def load_model_chkpoint(model, path):
    model = eval(model).load_from_checkpoint(path)
    model = model.cuda()
    model.eval()
    return model


def load_model(model_name, version):
    if version == -1:
        versions = os.listdir(
            f"/space/ariyanzarei/sorghum_segmentation/models/model_checkpoints/{model_name}/lightning_logs"
        )
        version = sorted(versions)[-1].split("_")[-1]

    path_all_checkpoints = f"/space/ariyanzarei/sorghum_segmentation/models/model_checkpoints/{model_name}/lightning_logs/version_{version}/checkpoints"
    path = sorted(os.listdir(path_all_checkpoints))[-1]
    print("Using Version ", version, " and ", path)

    model = load_model_chkpoint(model_name, os.path.join(path_all_checkpoints, path))
    return model


def main_ply(args, down_sample_n=8000):

    semantic_model = load_model(
        "SorghumPartNetSemantic", args.semantic_version
    ).double()
    instance_model = load_model(
        "SorghumPartNetInstance", args.instance_version
    ).double()

    for p in os.listdir(args.path):
        path = os.path.join(args.path, p)

        points_full, instance_labels, semantic_labels = load_real_ply_with_labels(path)

        downsample_indexes = random.sample(
            np.arange(0, points_full.shape[0]).tolist(),
            min(down_sample_n, points_full.shape[0]),
        )

        points = points_full[downsample_indexes]
        instance_labels = instance_labels[downsample_indexes]
        semantic_labels = semantic_labels[downsample_indexes]

        points = torch.tensor(points, dtype=torch.float64)
        points_full = torch.tensor(points_full, dtype=torch.float64)

        print(f":: Point cloud with {points_full.shape[0]} points loaded!")

        (
            downsampled_semantic_pcd,
            downsampled_instance_pcd,
            downsampled_semantics,
            downsampled_instance,
        ) = predict_downsampled(points, semantic_model, instance_model)

        instance_labels = instance_labels[downsampled_semantics == 1]
        metric_calculator = LeafMetrics(1)
        acc, precison, recal, f1 = metric_calculator(
            torch.tensor(downsampled_instance.astype(np.int32), dtype=torch.float64)
            .unsqueeze(0)
            .to(torch.device("cuda")),
            torch.tensor(instance_labels, dtype=torch.float64)
            .unsqueeze(0)
            .to(torch.device("cuda")),
        )
        print(acc, precison, recal, f1)

        if args.save:
            semantic_pcd, instance_pcd = pred_full_size(
                points_full, points[:, :3], downsampled_semantics, downsampled_instance
            )

            save_predicted(
                downsampled_semantic_pcd,
                f"/space/ariyanzarei/sorghum_segmentation/results/{p.replace('.ply','')}_semantic.ply",
            )
            save_predicted(
                downsampled_instance_pcd,
                f"/space/ariyanzarei/sorghum_segmentation/results/{p.replace('.ply','')}_instance.ply",
            )
            save_predicted(
                semantic_pcd,
                f"/space/ariyanzarei/sorghum_segmentation/results/{p.replace('.ply','')}_semantic_full.ply",
            )
            save_predicted(
                instance_pcd,
                f"/space/ariyanzarei/sorghum_segmentation/results/{p.replace('.ply','')}_instance_full.ply",
            )


def main():
    args = get_args()

    main_ply(args)


if __name__ == "__main__":
    main()
