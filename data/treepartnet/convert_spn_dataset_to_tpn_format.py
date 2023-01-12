import argparse
import h5py
import numpy as np
import os
import sys
import torch

sys.path.append("../..")
from models.treepartnet_utils import furthest_point_sample


def get_args():
    parser = argparse.ArgumentParser(
        description="Converting the SPNS original dataset in h5 format to TPN format to be run with the TPN model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i",
        "--input",
        help="The path to the directory containing training, validation and test h5 files. ",
        metavar="input",
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


def read_h5_dataset(path):
    with h5py.File(path) as f:
        data = np.array(f["points"])
        label = np.array(f["labels"])
    return data, label


def save_h5_dataset(dict_data, path):
    with h5py.File(path, "w") as f:
        for k in dict_data.keys():
            f.create_dataset(k, data=dict_data[k])


def furthest_point_sampling(points, n_initial_clusters=256):
    return furthest_point_sample(points, n_initial_clusters)


def calculate_initial_clustering(points):
    points = torch.from_numpy(points).type(torch.DoubleTensor)
    mins, _ = torch.min(points, axis=0)
    maxs, _ = torch.max(points, axis=0)
    mins = mins.unsqueeze(0)
    maxs = maxs.unsqueeze(0)
    points = (points - mins) / (maxs - mins) - 0.5
    points = points.cuda().unsqueeze(0).float()
    sample_idx = furthest_point_sample(points, 256).long().squeeze()
    samples = points[:, sample_idx, :]
    distances = torch.cdist(points, samples)
    _, nearest_idx = torch.topk(distances, 1, 2, False)
    initial_clusters = nearest_idx.squeeze().cpu()
    return (
        points.squeeze().cpu().numpy(),
        initial_clusters.numpy(),
        sample_idx.cpu().numpy(),
    )


def calculate_affinity_matrix(points, sample_idx, gt_label):
    points = torch.from_numpy(points).type(torch.DoubleTensor)
    gt_label = torch.from_numpy(gt_label).squeeze()
    sample_idx = torch.from_numpy(sample_idx)
    samples = points[sample_idx, :]
    distances = torch.cdist(samples, samples).squeeze()
    final_distances = distances.cpu()
    _, nearest_idx = torch.topk(final_distances, 3, 1, False)
    affinity_matrix = torch.eye(256)
    ind_new = (torch.arange(0, 256)[:, None], nearest_idx)
    affinity_matrix[ind_new] = 1
    sample_labels = gt_label[sample_idx].unsqueeze(0).unsqueeze(-1).float()
    label_similarity = torch.cdist(sample_labels, sample_labels)
    label_similarity = (label_similarity == 0).int().squeeze()
    affinity_matrix = torch.logical_and(label_similarity, affinity_matrix)
    return affinity_matrix.numpy()


def convert_format_single(points, labels):
    new_points, init_clusters, sample_idx = calculate_initial_clustering(points)
    affinity_matrix = calculate_affinity_matrix(points, sample_idx, labels)
    return new_points, init_clusters, labels, affinity_matrix, sample_idx


def convert_format_all(h5_input_path, h5_output_path):
    data, label = read_h5_dataset(h5_input_path)
    all_points = []
    all_local_clusters = []
    all_cluster_labels = []
    all_affinities = []
    all_local_context_idx = []

    n = data.shape[0]

    for i in range(n):
        print(f":: Processing point cloud {i+1}/{n}...")
        sys.stdout.flush()
        (
            points,
            local_clusters,
            cluster_labels,
            affinities,
            local_context_idx,
        ) = convert_format_single(data[i], label[i])
        all_points.append(points)
        all_local_clusters.append(local_clusters)
        all_cluster_labels.append(cluster_labels)
        all_affinities.append(affinities)
        all_local_context_idx.append(local_context_idx)

    all_points = np.stack(all_points, 0)
    all_local_clusters = np.stack(all_local_clusters, 0)
    all_cluster_labels = np.stack(all_cluster_labels, 0)
    all_affinities = np.stack(all_affinities, 0)
    all_local_context_idx = np.stack(all_local_context_idx, 0)

    data_dict = {
        "points": all_points,
        "local_clusters": all_local_clusters,
        "cluster_labels": all_cluster_labels,
        "affinities": all_affinities,
        "local_context_idx": all_local_context_idx,
    }

    save_h5_dataset(data_dict, h5_output_path)


def main():
    args = get_args()
    list_dataset_files = os.listdir(args.input)
    for dataset_file in list_dataset_files:
        if ".hdf5" not in dataset_file or "instance" not in dataset_file:
            continue
        input_dataset_file_path = os.path.join(args.input, dataset_file)
        output_dataset_file_path = os.path.join(args.output, dataset_file)
        convert_format_all(input_dataset_file_path, output_dataset_file_path)


main()
