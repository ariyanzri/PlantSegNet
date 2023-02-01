import argparse
import os
import h5py
import numpy as np
import json
import torch
import random
import glob
import sys

sys.path.append("..")
from data.load_raw_data import load_real_ply_with_labels
from models.nn_models import SorghumPartNetInstance
from models.utils import LeafMetrics, ClusterBasedMetrics
from data.utils import create_ply_pcd_from_points_with_labels

from sklearn.cluster import DBSCAN
from hyperopt import fmin, tpe, hp, Trials
from hyperopt.pyll import scope


def get_args():
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning script. This script runs the instance segmentation model on the validation sets of the given dataset and finds the best values of the hyperparameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-p",
        "--parameters",
        help="The path to the json file containing input arguments and hyperparameter values/ranges. ",
        metavar="parameters",
        required=True,
        type=str,
    )

    parser.add_argument(
        "-s",
        "--samples",
        help="Total number of samples to be drawn for hyperparameter tuning. ",
        metavar="samples",
        required=False,
        type=int,
        default=50,
    )

    parser.add_argument(
        "-i",
        "--iterations",
        help="Total number of iterations for the optimization. ",
        metavar="iterations",
        required=False,
        type=int,
        default=100,
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


def get_final_clusters(preds, DBSCAN_eps=1, DBSCAN_min_samples=10):
    preds = preds.cpu().detach().numpy().squeeze()
    clustering = DBSCAN(eps=DBSCAN_eps, min_samples=DBSCAN_min_samples).fit(preds)
    final_clusters = clustering.labels_
    return final_clusters


def get_preds(model, data, batch_size=4):
    predictions = []

    for i in range(0, data.shape[0], batch_size):
        print(
            f":: Predicting batch {int(i/batch_size)}/{int(data.shape[0]/batch_size)}",
            end="\r",
        )
        sys.stdout.flush()
        preds = model(data[i : i + batch_size])
        predictions.append(preds)
    return torch.cat(predictions)


def run_inference(predictions, label, DBSCAN_eps, DBSCAN_minpoints, device="cpu"):
    metric_calculator = LeafMetrics()
    clusterbased_metric_calculator = ClusterBasedMetrics([0.25, 0.5, 0.75], device)

    pointwise_accuracies = []
    pointwise_precisions = []
    pointwise_recals = []
    pointwise_f1s = []
    clusterbased_mean_coverages = []
    clusterbased_average_precisions = []
    clusterbased_average_recalls = []

    for i in range(predictions.shape[0]):
        preds = predictions[i : i + 1]

        try:
            pred_clusters = torch.tensor(
                get_final_clusters(preds, DBSCAN_eps, DBSCAN_minpoints)
            )
        except:
            continue

        acc, precison, recal, f1 = metric_calculator(
            pred_clusters.unsqueeze(0).unsqueeze(-1),
            label[i].unsqueeze(0).unsqueeze(-1),
        )

        pointwise_accuracies.append(acc)
        pointwise_precisions.append(precison)
        pointwise_recals.append(recal)
        pointwise_f1s.append(f1)

        clusterbased_metrics = clusterbased_metric_calculator(
            pred_clusters.squeeze(), label[i].squeeze()
        )
        clusterbased_mean_coverages.append(clusterbased_metrics["mean_coverage"])
        clusterbased_average_precisions.append(
            clusterbased_metrics["average_precision"]
        )
        clusterbased_average_recalls.append(clusterbased_metrics["average_recall"])

    mean_results_dic = {
        "pointwise_accuracy": np.mean(pointwise_accuracies),
        "pointwise_precision": np.mean(pointwise_precisions),
        "pointwise_recal": np.mean(pointwise_recals),
        "pointwise_f1": np.mean(pointwise_f1s),
        "clusterbased_mean_coverages": np.mean(clusterbased_mean_coverages),
        "clusterbased_average_precisions": np.mean(clusterbased_average_precisions),
        "clusterbased_average_recalls": np.mean(clusterbased_average_recalls),
    }

    return mean_results_dic


def objective_function(args):
    eps, minpoints, preds, label = args
    res = run_inference(preds, label, DBSCAN_eps=eps, DBSCAN_minpoints=minpoints)
    return -res["pointwise_accuracy"]


def save_results(path, full_results_dic, best_param):
    with open(os.path.join(path, "DBSCAN_full_results.json"), "w") as f:
        json.dump(full_results_dic, f)
    with open(os.path.join(path, "DBSCAN_best_param.json"), "w") as f:
        json.dump(best_param, f)


def load_params_dict(path):
    with open(path, "r") as f:
        params_dict = json.load(f)
    return params_dict


def main():
    args = get_args()
    train_param_dict = load_params_dict(args.parameters)

    output_base_path = os.path.join(
        "/space/ariyanzarei/sorghum_segmentation/results/training_logs",
        train_param_dict["model_name"],
        train_param_dict["dataset"],
    )

    output_dir = os.path.join(output_base_path, "HyperTuning")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_checkpoint_path = os.path.join(
        output_base_path, train_param_dict["experiment_id"], "checkpoints"
    )
    all_checkpoints = os.listdir(all_checkpoint_path)
    if len(all_checkpoints) != 1:
        print(":: Error reading the model checkpoint.")
        return
    best_checkpoint_path = os.path.join(all_checkpoint_path, all_checkpoints[0])
    model = load_model(train_param_dict["model_name"], best_checkpoint_path)

    if train_param_dict["dataset"] == "SPNS":
        data, label = load_data_h5(train_param_dict["val_data"], "points", "labels")
    elif train_param_dict["dataset"] == "TPN":
        data, label = load_data_h5(
            train_param_dict["val_data"], "points", "primitive_id"
        )
    else:
        print(":: Incorrect dataset name. ")
        return

    sample_indices = random.sample(
        np.arange(0, data.shape[0]).tolist(),
        min(args.samples, data.shape[0]),
    )

    data = data[sample_indices]
    label = label[sample_indices]

    device = "cpu"
    data = torch.Tensor(data).double().to(torch.device(device))
    label = torch.Tensor(label).float().squeeze().to(torch.device(device))
    model = model.to(torch.device(device))
    model.DGCNN_feature_space.device = device
    predictions = get_preds(model, data)

    trials = Trials()
    bestParams = fmin(
        fn=objective_function,
        space=[
            hp.uniform("eps", 0.05, 4),
            scope.int(hp.quniform("minpoints", 3, 15, q=1)),
            predictions,
            label,
        ],
        algo=tpe.suggest,
        max_evals=args.iterations,
        trials=trials,
    )

    full_res_dic = []
    for res in trials.trials:
        full_res_dic.append(
            {
                "eps": res["misc"]["vals"]["eps"],
                "minpoints": res["misc"]["vals"]["minpoints"],
                "accuracy": res["result"]["loss"],
            }
        )
    save_results(output_dir, full_res_dic, bestParams)


main()
