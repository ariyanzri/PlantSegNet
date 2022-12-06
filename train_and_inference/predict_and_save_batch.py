# from torch.multiprocessing import Pool, set_start_method
from multiprocessing import Pool
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys

sys.path.append("..")

from visualization.html_generator import create_html
from models.nn_models import *
from data.utils import create_ply_pcd_from_points_with_labels, distinct_colors
from data.load_raw_data import load_pcd_plyfile, load_ply_file_points
from predict_and_visualize import *
import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="Sorghum 3D part segmentation prediction script.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-v",
        "--version",
        help="The version of the model. If not determined, the latest version would be picked.",
        metavar="version",
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
        default=5,
    )

    parser.add_argument(
        "-p",
        "--path",
        help="Path to the dataset folder.",
        metavar="path",
        required=False,
        type=str,
        default="/space/ariyanzarei/sorghum_segmentation/dataset/real_data/2020-07-13/",
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Path to the output folder.",
        metavar="output",
        required=False,
        type=str,
        default="/space/ariyanzarei/sorghum_segmentation/results/2020-07-13",
    )

    parser.add_argument(
        "-t",
        "--type",
        help="Point cloud type, real or synthetic. 0 means synthetic and 1 means real.",
        metavar="type",
        required=False,
        type=int,
        default=1,
    )

    return parser.parse_args()


def generate_image(points, labels, is_instance, save_path):
    if is_instance:
        d_colors = distinct_colors(len(list(set(labels))))
        colors = np.zeros((labels.shape[0], 3))
        for i, l in enumerate(list(set(labels))):
            colors[labels == l, :] = d_colors[i]
    else:
        colors = np.column_stack((labels, labels, labels)).astype("float32")
        colors[colors[:, 0] == 0, :] = [0.3, 0.1, 0]
        colors[colors[:, 0] == 1, :] = [0, 0.7, 0]
        colors[colors[:, 0] == 2, :] = [0, 0, 0.7]

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, c=colors)
    plt.savefig(save_path)


def predict_and_generate_pcd(points, semantic_model, instance_model, device_name="cpu"):

    device = torch.device(device_name)
    semantic_model = semantic_model.to(device)
    semantic_model.DGCNN_semantic_segmentor.device = device_name
    instance_model = instance_model.to(device)
    instance_model.DGCNN_feature_space.device = device_name
    points = points.to(device)

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

    instance_points = points[pred_semantic_label_labels == 1]

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

    return (
        ply_semantic,
        ply_instance,
        pred_semantic_label_labels,
        pred_final_cluster,
    )


def predict_and_save_individual(arguments):
    args = arguments[0]
    pcd_name = arguments[1]

    semantic_model = load_model("SorghumPartNetSemantic", args.version).double()
    instance_model = load_model("SorghumPartNetInstance", args.version).double()

    try:

        path = os.path.join(args.path, pcd_name)
        outpath = os.path.join(args.output, pcd_name.replace(".ply", ""))
        if not os.path.exists(outpath):
            os.makedirs(outpath)

        if args.type == 0:
            pcd = load_pcd_plyfile(path)
            points = torch.tensor(pcd["points_full"], dtype=torch.float64)
        else:
            points, _, _ = load_ply_file_points(path)
            points = torch.tensor(points, dtype=torch.float64)

        print(f":: Point cloud with {points.shape[0]} points loaded!")

        (
            ply_semantic,
            ply_instance,
            semantic_pred_labels,
            instance_pred_labels,
        ) = predict_and_generate_pcd(points, semantic_model, instance_model)

        save_predicted(
            ply_semantic,
            os.path.join(outpath, "semantic.ply"),
        )
        save_predicted(
            ply_instance,
            os.path.join(outpath, "instance.ply"),
        )

        # Generate and save Images

        output_path = os.path.join(
            outpath, f"{pcd_name.replace('.ply', '')}_semantic.png"
        )
        generate_image(points, semantic_pred_labels, False, output_path)

        output_path = os.path.join(
            outpath, f"{pcd_name.replace('.ply', '')}_instance.png"
        )
        generate_image(
            points[semantic_pred_labels == 1], instance_pred_labels, True, output_path
        )

        print(f":: Generated images for {pcd_name}")
    except Exception as e:
        print(":: Error happened --> " + str(e))


def predict_batch(args):

    list_pcd_files = os.listdir(args.path)
    n = len(list_pcd_files)

    arguments = []
    for i, pcd_name in enumerate(list_pcd_files):
        # arguments.append((args, pcd_name))
        predict_and_save_individual((args, pcd_name))
        if i >= 10:
            break

        # print(f"------------- ({i+1}/{n}) -------------")
        # path = os.path.join(args.path, pcd_name)

        # outpath = os.path.join(args.output, pcd_name.replace(".ply", ""))
        # if not os.path.exists(outpath):
        #     os.makedirs(outpath)

        # if args.type == 0:
        #     pcd = load_pcd_plyfile(path)
        #     points = torch.tensor(pcd["points"], dtype=torch.float64)
        #     points_full = torch.tensor(pcd["points_full"], dtype=torch.float64)
        # else:
        #     points_full, points = load_ply_file_points(path)
        #     points = torch.tensor(points, dtype=torch.float64)
        #     points_full = torch.tensor(points_full, dtype=torch.float64)

        # print(f":: Point cloud with {points_full.shape[0]} points loaded!")

        # (
        #     downsampled_semantic_pcd,
        #     downsampled_instance_pcd,
        #     downsampled_semantics,
        #     downsampled_instance,
        # ) = predict_downsampled(points, semantic_model, instance_model)

        # semantic_pcd, instance_pcd = pred_full_size(
        #     points_full, points, downsampled_semantics, downsampled_instance
        # )

        # save_predicted(
        #     downsampled_semantic_pcd,
        #     os.path.join(outpath, "semantic.ply"),
        # )
        # save_predicted(
        #     downsampled_instance_pcd,
        #     os.path.join(outpath, "instance.ply"),
        # )
        # save_predicted(semantic_pcd, os.path.join(outpath, "semantic_full.ply"))
        # save_predicted(instance_pcd, os.path.join(outpath, "instance_full.ply"))

    # set_start_method("spawn")
    # with Pool(5) as pool:
    #     pool.map(predict_and_save_individual, arguments)


def main():
    args = get_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    predict_batch(args)
    create_html(args.output, args.output, "visualization")
    # create_html(args.output, args.output, "visualization")


if __name__ == "__main__":
    main()
