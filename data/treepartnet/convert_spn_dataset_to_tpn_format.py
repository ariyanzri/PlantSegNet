import argparse
import h5py
import numpy as np
import os
import sys

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


def calculate_initial_clustering():
    # for each point, find nearest point in furthest samples
    # and assign the index of that sample as the initial cluster
    # value
    pass


def calculate_affinity_matrix():
    # for each furthest sample point, find its three closest points
    # with the same branch id among the furthest sample points.
    # These are considered 1 in the affinity matrix and the rest
    # will be zero
    pass


def convert_format_single(points, labels):
    pass


def convert_format_all(h5_input_path, h5_output_path):
    input_data = read_h5_dataset(h5_input_path)
    all_points = []
    all_local_clusters = []
    all_cluster_labels = []
    all_affinities = []
    all_local_context_idx = []

    for i in range(input_data["points"].shape[0]):
        (
            points,
            local_clusters,
            cluster_labels,
            affinities,
            local_context_idx,
        ) = convert_format_single(input_data["points"][i], input_data["labels"][i])
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
        if ".hdf5" not in dataset_file:
            continue
        input_dataset_file_path = os.path.join(args.input, dataset_file)
        output_dataset_file_path = os.path.join(args.output, dataset_file)
        convert_format_all(input_dataset_file_path, output_dataset_file_path)


main()
