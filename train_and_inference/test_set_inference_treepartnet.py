import argparse
import os
import h5py
import numpy as np
import random
import json
import torch
import open3d as o3d
import sys

sys.path.append("..")
from data.load_raw_data import load_real_ply_with_labels
from models.nn_models import TreePartNet
from models.utils import LeafMetrics, AveragePrecision
from data.utils import create_ply_pcd_from_points_with_labels

from sklearn.cluster import DBSCAN


def get_args():
    parser = argparse.ArgumentParser(
        description="Test set inference script for the TreePartNet model. This script runs the instance segmentation model on the test sets of the given dataset. So far it only works with the TPN dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i",
        "--input",
        help="The path to the test h5 file or directory containing the test point clouds. ",
        metavar="input",
        required=True,
        type=str,
    )

    parser.add_argument(
        "-m",
        "--model",
        help="The path to the model checkpoint. ",
        metavar="model",
        required=True,
        type=str,
    )

    parser.add_argument(
        "-d",
        "--dataset",
        help="The name of the dataset. It can only be TPN or SPNS at this point. ",
        metavar="dataset",
        required=True,
        type=str,
    )

    parser.add_argument(
        "-o",
        "--output",
        help="The path to the directory in which the output files will be saved. ",
        metavar="output",
        required=True,
        type=str,
    )

    return parser.parse_args()


def load_model(model, path):
    model = eval(model).load_from_checkpoint(path)
    model.eval()
    return model


def load_data_h5(path, point_key, label_key):
    with h5py.File(path) as f:
        data = np.array(f[point_key])
        label = np.array(f[label_key])
    return data, label


def get_final_clusters(preds, merge_similar_clusters=False, merge_threshold=0):
    try:
        pred_cluster = preds[0]
        pred_aff = preds[1].squeeze()
        pred_samples_idx = preds[2].squeeze()
        pred_cluster = torch.argmax(pred_cluster, 1).squeeze()

        if not merge_similar_clusters:
            return pred_cluster.cpu()

        pred_aff = (pred_aff >= merge_threshold).int().squeeze()
        pred_final_cluster = pred_cluster.cpu()

        for i in range(256):
            for j, v in enumerate(pred_aff[i]):
                if v == 1:
                    pred_final_cluster[
                        pred_final_cluster == pred_final_cluster[pred_samples_idx[j]]
                    ] = pred_final_cluster[pred_samples_idx[i]]

        return pred_final_cluster
    except Exception as e:
        print(e)
        return None


def run_inference(model, data, label, best_params):
    data = torch.from_numpy(data).type(torch.FloatTensor).cuda()
    label = torch.Tensor(label).float().squeeze().cuda()
    model = model.cuda()

    metric_calculator = LeafMetrics("cuda")
    ap_calculator = AveragePrecision(0.25, "cuda")

    best_acc_value = 0
    best_acc_preds = None
    best_acc_index = -1
    worst_acc_value = 100
    worst_acc_preds = None
    worst_acc_index = -1

    pointwise_accuracies = []
    pointwise_precisions = []
    pointwise_recals = []
    pointwise_f1s = []
    aps = []

    final_predictions = []

    for i in range(data.shape[0]):
        if i == 10:
            break
        preds = model(data[i : i + 1])
        pred_clusters = get_final_clusters(
            preds, best_params["merge_similar_clusters"], best_params["merge_threshold"]
        )
        if pred_clusters is None:
            continue

        final_predictions.append(pred_clusters)
        pred_clusters = torch.tensor(pred_clusters)

        acc, precison, recal, f1 = metric_calculator(
            pred_clusters.unsqueeze(0).unsqueeze(-1).cuda(),
            label[i].unsqueeze(0).unsqueeze(-1),
        )

        pointwise_accuracies.append(acc)
        pointwise_precisions.append(precison)
        pointwise_recals.append(recal)
        pointwise_f1s.append(f1)

        if acc > best_acc_value:
            best_acc_value = acc
            best_acc_index = i
            best_acc_preds = pred_clusters

        if acc < worst_acc_value:
            worst_acc_value = acc
            worst_acc_index = i
            worst_acc_preds = pred_clusters

        ap = ap_calculator(pred_clusters.squeeze(), label[i].squeeze())
        aps.append(ap)

        print(f":: Instance {i}/{data.shape[0]}= Accuracy: {acc} - AP: {ap}")
        sys.stdout.flush()

    full_results_dic = {
        "pointwise_accuracies": pointwise_accuracies,
        "pointwise_precisions": pointwise_precisions,
        "pointwise_recals": pointwise_recals,
        "pointwise_f1s": pointwise_f1s,
        "average_precisions": aps,
    }

    mean_results_dic = {
        "pointwise_accuracy": np.mean(pointwise_accuracies),
        "pointwise_precision": np.mean(pointwise_precisions),
        "pointwise_recal": np.mean(pointwise_recals),
        "pointwise_f1": np.mean(pointwise_f1s),
        "average_precision": np.mean(aps),
    }

    best_example_ply = create_ply_pcd_from_points_with_labels(
        data[best_acc_index].squeeze().cpu().numpy(),
        best_acc_preds.squeeze().cpu().numpy(),
    )
    worst_example_ply = create_ply_pcd_from_points_with_labels(
        data[worst_acc_index].squeeze().cpu().numpy(),
        worst_acc_preds.squeeze().cpu().numpy(),
    )

    final_predictions = np.stack(final_predictions, 0)

    return (
        full_results_dic,
        mean_results_dic,
        best_example_ply,
        worst_example_ply,
        final_predictions,
    )


def save_results(
    path,
    full_results_dic,
    mean_results_dic,
    best_example_ply,
    worst_example_ply,
    final_predictions,
):
    o3d.io.write_point_cloud(os.path.join(path, "best_example.ply"), best_example_ply)
    o3d.io.write_point_cloud(os.path.join(path, "worst_example.ply"), worst_example_ply)
    with open(os.path.join(path, "full_results.json"), "w") as f:
        json.dump(full_results_dic, f)
    with open(os.path.join(path, "mean_results.json"), "w") as f:
        json.dump(mean_results_dic, f)
    with h5py.File(os.path.join(path, "predictions.hdf5"), "w") as f:
        f.create_dataset("preds", data=final_predictions)


def main():
    args = get_args()

    output_dir = os.path.join(args.output, args.dataset)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = load_model("TreePartNet", args.model)

    if args.dataset == "TPN":
        data, label = load_data_h5(args.input, "points", "primitive_id")
    elif args.dataset == "SPNS":
        data, label = load_data_h5(args.input, "points", "cluster_labels")
    else:
        print(":: Incorrect dataset name. ")
        return

    print(f":: Starting the inference")
    sys.stdout.flush()

    best_params = {"merge_similar_clusters": False, "merge_threshold": 0.5}

    (
        full_results_dic,
        mean_results_dic,
        best_example_ply,
        worst_example_ply,
        final_predictions,
    ) = run_inference(model, data, label, best_params)
    save_results(
        output_dir,
        full_results_dic,
        mean_results_dic,
        best_example_ply,
        worst_example_ply,
        final_predictions,
    )


main()

"""
Running argument samples for all datasets:
TPN: nohup python test_set_inference_treepartnet.py -i /space/ariyanzarei/sorghum_segmentation/dataset/TreePartNetData/tree_labeled_test.hdf5 -m /space/ariyanzarei/sorghum_segmentation/models/other_datasets_model_checkpoints/TPN/TreePartNet/lightning_logs/version_0/checkpoints/epoch\=9-step\=8809.ckpt -d TPN -o /space/ariyanzarei/sorghum_segmentation/results/TPN_model_test_set/ &>nohup_test.out&
SPNS: python test_set_inference_treepartnet.py -i /space/ariyanzarei/sorghum_segmentation/dataset/synthetic/2022-12-26/h5_tpn/instance_segmentation_validation.hdf5 -m /space/ariyanzarei/sorghum_segmentation/models/other_datasets_model_checkpoints/SPN/TreePartNet/lightning_logs/version_0/checkpoints/epoch\=8-step\=86399.ckpt -d SPNS -o /space/ariyanzarei/sorghum_segmentation/results/TPN_model_results
"""
