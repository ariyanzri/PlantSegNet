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
from sklearn.cluster import DBSCAN
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


def predict_and_generate_pcd(
    points, instance_points, semantic_model, instance_model, dist=5, device_name="cpu"
):

    device = torch.device(device_name)
    semantic_model = semantic_model.to(device)
    semantic_model.DGCNN_semantic_segmentor.device = device_name
    instance_model = instance_model.to(device)
    instance_model.DGCNN_feature_space.device = device_name
    points = points.to(device)
    instance_points = instance_points.to(device)

    if (
        "use_normals" in semantic_model.hparams
        and semantic_model.hparams["use_normals"]
    ):
        pred_semantic_label = semantic_model(torch.unsqueeze(points, dim=0).to(device))
    else:
        pred_semantic_label = semantic_model(
            torch.unsqueeze(points[:, :3], dim=0).to(device)
        )

    pred_semantic_label = F.softmax(pred_semantic_label, dim=1)
    pred_semantic_label = pred_semantic_label[0].cpu().detach().numpy().T
    pred_semantic_label_labels = np.argmax(pred_semantic_label, 1)

    if (
        "use_normals" in instance_model.hparams
        and instance_model.hparams["use_normals"]
    ):
        pred_instance_features = instance_model(
            torch.unsqueeze(instance_points, dim=0).to(device)
        )
    else:
        pred_instance_features = instance_model(
            torch.unsqueeze(instance_points[:, :3], dim=0).to(device)
        )

    pred_instance_features = pred_instance_features.cpu().detach().numpy().squeeze()
    clustering = DBSCAN(eps=1, min_samples=10).fit(pred_instance_features)
    pred_final_cluster = clustering.labels_

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


def save_predicted(ply_pcd, path):
    o3d.io.write_point_cloud(path, ply_pcd)


def load_model_chkpoint(model, path):
    model = eval(model).load_from_checkpoint(path)
    # model = model.cuda()
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


def main_ply(args):

    semantic_model = load_model(
        "SorghumPartNetSemantic", args.semantic_version
    ).double()
    instance_model = load_model(
        "SorghumPartNetInstance", args.instance_version
    ).double()

    for p in os.listdir(args.path):
        path = os.path.join(args.path, p)

        points, instance_labels, semantic_labels = load_real_ply_with_labels(path)

        instance_points = points[semantic_labels == 1]
        instance_labels = instance_labels[semantic_labels == 1]

        points = torch.tensor(points, dtype=torch.float64)
        instance_points = torch.tensor(instance_points, dtype=torch.float64)

        print(f":: Point cloud with {points.shape[0]} points loaded!")

        (
            semantic_pcd,
            instance_pcd,
            semantic_predictions,
            instance_predictions,
        ) = predict_and_generate_pcd(
            points, instance_points, semantic_model, instance_model
        )

        # instance_labels = instance_labels[downsampled_semantics == 1]
        metric_calculator = LeafMetrics(1)
        acc, precison, recal, f1 = metric_calculator(
            torch.tensor(instance_predictions.astype(np.int32), dtype=torch.float64)
            .unsqueeze(0)
            .to(torch.device("cuda")),
            torch.tensor(instance_labels, dtype=torch.float64)
            .unsqueeze(0)
            .to(torch.device("cuda")),
        )

        print(acc, precison, recal, f1)

        if args.save:
            save_predicted(
                semantic_pcd,
                f"/space/ariyanzarei/sorghum_segmentation/results/{p.replace('.ply','')}_semantic.ply",
            )
            save_predicted(
                instance_pcd,
                f"/space/ariyanzarei/sorghum_segmentation/results/{p.replace('.ply','')}_instance.ply",
            )


def main():
    args = get_args()

    main_ply(args)


if __name__ == "__main__":
    main()
