import argparse
import os
import h5py
import numpy as np
import random
import json
import torch
import torch.nn.functional as F
import sys

sys.path.append("..")
from data.load_raw_data import load_real_ply_with_labels
from models.nn_models import SorghumPartNetInstance, SorghumPartNetSemantic, TreePartNet
from models.utils import LeafMetrics, ClusterBasedMetrics, SemanticMetrics
from data.utils import create_ply_pcd_from_points_with_labels

from sklearn.cluster import DBSCAN


def get_args():
    parser = argparse.ArgumentParser(
        description="Test set inference script. This script runs our instance / semantic segmentation models as well as TreePartNet on the test sets of the given dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-e",
        "--experiment",
        help="The path to the experiment json file for which the model checkpoint will be loaded and tests will be run according to the test set path in the json.",
        metavar="experiment",
        required=True,
        type=str,
    )

    parser.add_argument(
        "-o",
        "--output",
        help="The path to the root directory of results",
        metavar="output",
        required=False,
        default="/speedy/ariyanzarei/sorghum_segmentation/results",
        type=str,
    )

    return parser.parse_args()


def load_json(path):
    with open(path, "r") as f:
        params_dict = json.load(f)
    return params_dict


def load_model(model_name, path, device="cpu"):
    checkpoint_names = os.listdir(path)
    if len(checkpoint_names) != 1:
        print(
            ":: Error: Number of checkpoints is zero or more than one. Please keep only one checkpoint. "
        )
        return None

    path = os.path.join(path, checkpoint_names[0])
    model = eval(model_name).load_from_checkpoint(path)
    model.eval()
    model = model.to(torch.device(device))

    if model_name == "SorghumPartNetInstance":
        model.DGCNN_feature_space.device = device
    elif model_name == "SorghumPartNetSemantic":
        model.DGCNN_semantic_segmentor.device = device

    return model


def load_data_h5(path, point_key, label_key):
    with h5py.File(path) as f:
        data = np.array(f[point_key])
        label = np.array(f[label_key])
    return data, label


def load_data_original_size(h5_path, path, point_key, label_key, index):
    with h5py.File(h5_path) as f:
        name = f["names"][index]
        print(name)


def load_data_directory(path, model_name):
    is_instance = model_name != "SorghumPartNetSemantic"

    data = []
    labels = []
    min_shape = sys.maxsize

    for p in os.listdir(path):
        file_path = os.path.join(path, p)
        points, instance_labels, semantic_labels = load_real_ply_with_labels(file_path)
        instance_points = points[semantic_labels == 1]
        instance_labels = instance_labels[semantic_labels == 1]
        if is_instance:
            data.append(instance_points)
            labels.append(instance_labels)
            if instance_labels.shape[0] < min_shape:
                min_shape = instance_labels.shape[0]
        else:
            data.append(points)
            labels.append(semantic_labels)
            if semantic_labels.shape[0] < min_shape:
                min_shape = semantic_labels.shape[0]

    resized_data = []
    resized_labels = []
    random.seed(10)

    for i, datum in enumerate(data):
        label = labels[i]
        downsample_indexes = random.sample(
            np.arange(0, datum.shape[0]).tolist(),
            min_shape,
        )
        datum = datum[downsample_indexes]
        label = label[downsample_indexes]
        resized_data.append(datum)
        resized_labels.append(label)

    resized_data = np.stack(resized_data)
    resized_labels = np.stack(resized_labels)

    return resized_data, resized_labels


def get_final_clusters_SPN(preds, DBSCAN_eps=1, DBSCAN_min_samples=10):
    try:
        preds = preds.cpu().detach().numpy().squeeze()
        clustering = DBSCAN(eps=DBSCAN_eps, min_samples=DBSCAN_min_samples).fit(preds)
        final_clusters = clustering.labels_
        return final_clusters
    except Exception as e:
        print(e)
        return None


def get_final_clusters_TPN(preds, merge_similar_clusters=False, merge_threshold=0):
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


def get_final_clusters(args):
    model_name = args[0]
    if model_name == "SorghumPartNetInstance":
        return get_final_clusters_SPN(*args[1:])
    elif model_name == "TreePartNet":
        return get_final_clusters_TPN(*args[1:])


def run_inference_semantic(model, data, label):
    data = torch.from_numpy(data).type(torch.DoubleTensor)
    label = torch.Tensor(label).float().squeeze().cpu()

    semantic_metric_calculator = SemanticMetrics()

    accuracies = []

    for i in range(data.shape[0]):
        # if i == 10:
        #     break
        preds = model(data[i : i + 1])
        preds = F.softmax(preds, dim=1)
        preds = preds.squeeze().cpu().detach().numpy().T
        preds = np.argmax(preds, 1)

        acc = semantic_metric_calculator(torch.Tensor(preds), label[i].squeeze())
        accuracies.append(acc)

        print(f":: Instance {i}/{data.shape[0]}= Accuracy: {acc}")

        sys.stdout.flush()

    full_results_dic = {
        "accuracies": accuracies,
    }

    mean_results_dic = {"accuracy": np.mean(accuracies)}

    return (full_results_dic, mean_results_dic)


def run_inference_instance(model_name, model, data, label, best_params, device="cpu"):
    if model_name == "TreePartNet":
        data = torch.from_numpy(data).type(torch.FloatTensor).to(torch.device(device))
    else:
        data = torch.from_numpy(data).type(torch.DoubleTensor).to(torch.device(device))
    label = torch.Tensor(label).float().squeeze().to(torch.device(device))

    pointwise_metric_calculator = LeafMetrics(device)
    clusterbased_metric_calculator = ClusterBasedMetrics([0.25, 0.5, 0.75], device)

    pointwise_accuracies = []
    pointwise_precisions = []
    pointwise_recalls = []
    pointwise_f1s = []
    clusterbased_mean_coverages = []
    clusterbased_average_precisions = []
    clusterbased_average_recalls = []

    if model_name == "SorghumPartNetInstance":
        eps = (
            best_params["eps"]
            if best_params is not None and "eps" in best_params
            else None
        )
        minpoints = (
            best_params["minpoints"]
            if best_params is not None and "minpoints" in best_params
            else None
        )
        hparams = (eps, minpoints)
    elif model_name == "TreePartNet":
        merge_similar_clusters = (
            best_params["merge_similar_clusters"]
            if best_params is not None and "merge_similar_clusters" in best_params
            else None
        )
        merge_threshold = (
            best_params["merge_threshold"]
            if best_params is not None and "merge_threshold" in best_params
            else None
        )
        hparams = (merge_similar_clusters, merge_threshold)

    for i in range(data.shape[0]):
        # if i == 10:
        #     break
        preds = model(data[i : i + 1])
        pred_clusters = get_final_clusters(
            (model_name, preds) if hparams[0] is None else (model_name, preds) + hparams
        )
        if pred_clusters is None:
            continue

        pred_clusters = torch.tensor(pred_clusters).to(torch.device(device))

        acc, precison, recall, f1 = pointwise_metric_calculator(
            pred_clusters.unsqueeze(0).unsqueeze(-1),
            label[i].unsqueeze(0).unsqueeze(-1),
        )

        pointwise_accuracies.append(acc)
        pointwise_precisions.append(precison)
        pointwise_recalls.append(recall)
        pointwise_f1s.append(f1)

        clusterbased_metrics = clusterbased_metric_calculator(
            pred_clusters.squeeze(), label[i].squeeze()
        )

        clusterbased_mean_coverages.append(clusterbased_metrics["mean_coverage"])
        clusterbased_average_precisions.append(
            clusterbased_metrics["average_precision"]
        )
        clusterbased_average_recalls.append(clusterbased_metrics["average_recall"])

        print(
            f":: Instance {i}/{data.shape[0]}= Accuracy: {acc} - mCov: {clusterbased_metrics['mean_coverage']} - Average Precision: {clusterbased_metrics['average_precision']} - Average Recall: {clusterbased_metrics['average_recall']}"
        )

        sys.stdout.flush()

    full_results_dic = {
        "pointwise_accuracies": pointwise_accuracies,
        "pointwise_precisions": pointwise_precisions,
        "pointwise_recalls": pointwise_recalls,
        "pointwise_f1s": pointwise_f1s,
        "clusterbased_mean_coverages": clusterbased_mean_coverages,
        "clusterbased_average_precisions": clusterbased_average_precisions,
        "clusterbased_average_recalls": clusterbased_average_recalls,
    }

    mean_results_dic = {
        "pointwise_accuracy": np.mean(pointwise_accuracies),
        "pointwise_precision": np.mean(pointwise_precisions),
        "pointwise_recall": np.mean(pointwise_recalls),
        "pointwise_f1": np.mean(pointwise_f1s),
        "clusterbased_mean_coverage": np.mean(clusterbased_mean_coverages),
        "clusterbased_average_precision": np.mean(clusterbased_average_precisions),
        "clusterbased_average_recall": np.mean(clusterbased_average_recalls),
    }

    return (full_results_dic, mean_results_dic)


def run_inference(args):
    model_name = args[0]
    if model_name in ["SorghumPartNetInstance", "TreePartNet"]:
        return run_inference_instance(*args)
    else:
        return run_inference_semantic(*args[1:-2])


def save_results(path, full_results_dic, mean_results_dic):
    with open(os.path.join(path, "full_results.json"), "w") as f:
        json.dump(full_results_dic, f)
    with open(os.path.join(path, "mean_results.json"), "w") as f:
        json.dump(mean_results_dic, f)


def main():
    args = get_args()
    experiment_params = load_json(args.experiment)

    dataset_name = experiment_params["dataset"]
    model_name = experiment_params["model_name"]
    experiment_id = experiment_params["experiment_id"]

    device = "cuda" if model_name == "TreePartNet" else "cpu"

    model_path = os.path.join(
        args.output,
        "training_logs",
        model_name,
        dataset_name,
        experiment_id,
        "checkpoints",
    )
    model = load_model(model_name, model_path, device)

    hyperparameter_path = os.path.join(
        args.output,
        "hparam_tuning_logs",
        model_name,
        dataset_name,
        experiment_id,
        "DBSCAN_best_param.json",
    )
    if os.path.exists(hyperparameter_path):
        best_hparams = load_json(hyperparameter_path)
        print(":: Hyperparameter file found. Loaded the parameters as: ")
        print(best_hparams)
    else:
        best_hparams = None

    if "real_data" in experiment_params:
        data, label = load_data_directory(experiment_params["real_data"], model_name)
        print(f":: Starting the inference on the real data...")
        sys.stdout.flush()
        (full_results_dic, mean_results_dic) = run_inference(
            (model_name, model, data, label, best_hparams, device)
        )

        output_dir = os.path.join(
            args.output,
            "inference_logs",
            model_name,
            dataset_name,
            experiment_id,
            "real_data",
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_results(output_dir, full_results_dic, mean_results_dic)

    if "test_data" in experiment_params:
        dataset_path = experiment_params["test_data"]
        if dataset_name == "SPNS" and model_name == "SorghumPartNetInstance":
            data, label = load_data_h5(dataset_path, "points", "labels")
        elif dataset_name == "SPNS" and model_name == "TreePartNet":
            data, label = load_data_h5(dataset_path, "points", "cluster_labels")
        elif dataset_name == "TPN":
            data, label = load_data_h5(dataset_path, "points", "primitive_id")
        print(f":: Starting the inference on the test dataset...")
        sys.stdout.flush()
        (
            full_results_dic,
            mean_results_dic,
        ) = run_inference((model_name, model, data, label, best_hparams, device))

        output_dir = os.path.join(
            args.output,
            "inference_logs",
            model_name,
            dataset_name,
            experiment_id,
            "test_set",
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_results(
            output_dir,
            full_results_dic,
            mean_results_dic,
        )


if __name__ == "__main__":
    main()
