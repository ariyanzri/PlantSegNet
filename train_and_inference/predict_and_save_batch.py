from inspect import getargs
import multiprocessing
from torch.multiprocessing import Pool, set_start_method
import os
from symbol import argument
import numpy as np
import torch
import open3d as o3d
import torch.nn.functional as F
import sys

sys.path.append("..")


from visualization.gif_generator import generate_bulk_gif, save_gif
from visualization.html_generator import create_html
from models.nn_models import *
from models.datasets import SorghumDataset
from data.utils import create_ply_pcd_from_points_with_labels
from data.load_raw_data import load_pcd_plyfile, load_ply_file_points
from scipy.spatial.distance import cdist
from scipy import stats
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
        default="/space/ariyanzarei/sorghum_segmentation/dataset/2022-03-10/PointCloud",
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Path to the output folder.",
        metavar="output",
        required=False,
        type=str,
        default="/space/ariyanzarei/sorghum_segmentation/results",
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


def predict_and_save_individual(arguments):
    args = arguments[0]
    pcd_name = arguments[1]

    try:
        semantic_model = load_model("SorghumPartNetSemantic", args.version).double()
        instance_model = load_model("SorghumPartNetInstance", args.version).double()

        path = os.path.join(args.path, pcd_name)
        outpath = os.path.join(args.output, pcd_name.replace(".ply", ""))
        if not os.path.exists(outpath):
            os.makedirs(outpath)

        if args.type == 0:
            pcd = load_pcd_plyfile(path)
            points = torch.tensor(pcd["points"], dtype=torch.float64)
            points_full = torch.tensor(pcd["points_full"], dtype=torch.float64)
        else:
            points_full, points = load_ply_file_points(path)
            points = torch.tensor(points, dtype=torch.float64)
            points_full = torch.tensor(points_full, dtype=torch.float64)

        print(f":: Point cloud with {points_full.shape[0]} points loaded!")

        (
            downsampled_semantic_pcd,
            downsampled_instance_pcd,
            downsampled_semantics,
            downsampled_instance,
        ) = predict_downsampled(points, semantic_model, instance_model)

        semantic_pcd, instance_pcd = pred_full_size(
            points_full, points, downsampled_semantics, downsampled_instance
        )

        save_predicted(
            downsampled_semantic_pcd,
            os.path.join(outpath, "semantic.ply"),
        )
        save_predicted(
            downsampled_instance_pcd,
            os.path.join(outpath, "instance.ply"),
        )
        save_predicted(semantic_pcd, os.path.join(outpath, "semantic_full.ply"))
        save_predicted(instance_pcd, os.path.join(outpath, "instance_full.ply"))

        # Generate and save GIFs

        input_path = os.path.join(outpath, "semantic.ply")
        output_path = os.path.join(
            outpath, f"{pcd_name.replace('.ply', '')}_semantic.gif"
        )
        save_gif(input_path, output_path)

        input_path = os.path.join(outpath, "instance.ply")
        output_path = os.path.join(
            outpath, f"{pcd_name.replace('.ply', '')}_instance.gif"
        )
        save_gif(input_path, output_path)

        print(f":: Generated gif for {pcd_name}")
    except Exception as e:
        print(":: Error happened --> " + e)


def predict_batch(args):

    # semantic_model = load_model("SorghumPartNetSemantic", args.version).double()
    # instance_model = load_model("SorghumPartNetInstance", args.version).double()

    list_pcd_files = os.listdir(args.path)
    n = len(list_pcd_files)

    arguments = []
    for i, pcd_name in enumerate(list_pcd_files):
        arguments.append((args, pcd_name))

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

    set_start_method("spawn")
    with Pool(5) as pool:
        pool.map(predict_and_save_individual, arguments)


def main():
    args = get_args()

    # predict_batch(args)
    create_html(args.output, args.output, "visualization")
    create_html(args.output, args.output, "visualization")


if __name__ == "__main__":
    main()
