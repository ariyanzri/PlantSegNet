import h5py
import os
import numpy as np
import sys
sys.path.append("../..")
from SorghumPartNet.data.load_raw_data import load_pcd_plyfile
import k3d
import open3d as o3d
import math
from sklearn.model_selection import train_test_split
import argparse


def filter_points(leaf_dataset, key, idx):
    leaf_dataset["points"][key] = leaf_dataset["points"][key][idx]
    leaf_dataset["leaf_index"][key] = leaf_dataset["leaf_index"][key][idx]


def filter_samples(leaf_dataset, keys):

    # The tolist at the end of these lines is to allow the h5py conversion to succeed
    leaf_dataset["points"] = (np.array(leaf_dataset["points"])[keys]).tolist()
    leaf_dataset["leaf_index"] = (np.array(leaf_dataset["leaf_index"])[keys]).tolist()
    leaf_dataset["leaf_count"] = (np.array(leaf_dataset["leaf_count"])[keys]).tolist()
    leaf_dataset["_plant_index"] = ((np.array(leaf_dataset["_plant_index"]).astype(np.int8))[keys]).tolist()


def apply_segment_error(leaf_dataset, key, segment_error_rate=.5):
    """
    Given a leaf dataset, modify in-place to simulate segmentation errors.
    """

    points = leaf_dataset["points"][key]
    point_count = points.shape[0]
    target_count = math.ceil(point_count * (1-segment_error_rate))

    focal_idx = np.random.randint(0, point_count)
    focal_point = points[focal_idx]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)    
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    [k, idx, _] =  pcd_tree.search_knn_vector_3d(focal_point, target_count)

    filter_points(leaf_dataset, key, idx)


def add_merged(leaf_dataset, idx_a, idx_b):
    """
    Given indices of two leafs in dataset create a new entry from merger of pair
    and append to leaf_dataset.
    """

    paired_points = np.concatenate((leaf_dataset["points"][idx_a], leaf_dataset["points"][idx_b]))
    paired_leaf_indices = np.concatenate(
        (leaf_dataset["leaf_index"][idx_a], leaf_dataset["leaf_index"][idx_b]))
    paired_leaf_count = 2

    assert leaf_dataset["_plant_index"][idx_a] == leaf_dataset["_plant_index"][idx_b]

    leaf_dataset["points"].append(paired_points)
    leaf_dataset["leaf_index"].append(paired_leaf_indices)
    leaf_dataset["leaf_count"].append(paired_leaf_count)
    leaf_dataset["_plant_index"].append(leaf_dataset["_plant_index"][idx_a])


def apply_down_sample(leaf_dataset, key, down_sample_n):
    """
    Given index into leaf dataset downsample number of points at that index
    """
    points = leaf_dataset["points"][key]
    point_count = points.shape[0]

    idx = np.random.choice(point_count, down_sample_n)
    filter_points(leaf_dataset, key, idx)


def generate_leaf_dataset(
    plant_processed_training_path,
    down_sample_n=80, # do we need this?
    segment_error=True,
    segment_error_pop_percent=1,
    segment_error_rate=0.5,
    total_sample_n=200,
    centroid_dist_threshold=0.5
    ):
    """
    Given path to multiple full plant ".ply" files generate a single vs paired leaf dataset with specified
    options.
    """

    double_leaf_sample_n = int(total_sample_n * .5)

    plant_dataset = h5py.File(plant_processed_training_path, 'r')

    # define empty output dict
    leaf_dataset = {
        'points': [],
        'leaf_index': [],
        'leaf_count': [], 
        '_plant_index': [],
        }

    # we need to build a partial plant dataset to derive the leaf data

    plant_dataset_additional = {
        'leaf_ref_index': [],
        'max_leaves': [],
        'leaf_centroids': [],
    }

    plant_count = plant_dataset["points"].shape[0]

    leaf_total_count = 0

    # for each plant in the training dataset
    for i in range(plant_count):
        print(f"[Info] Processing plant #{i}")

        plant_points = plant_dataset['points'][i]
        plant_is_focal = plant_dataset['is_focal_plant'][i]
        plant_leaf_index = plant_dataset['leaf_index'][i]
        plant_index = plant_dataset['plant_index'][i]
        plant_name = plant_dataset['names'][i]

        plant_points[plant_is_focal == 0] == np.NaN
        plant_leaf_index[plant_is_focal == 0] == np.NaN

        leaf_indices = np.unique(plant_leaf_index)
        max_leaves = leaf_indices.shape[0]

        # init 3d array of leaf centroids from which we can sample leaf data
        leaf_centroids = []        

        # index in leaf_dataset of a particular leaf
        leaf_ref_indices = []

        leaf_count = 0
        for j, leaf_index in enumerate(leaf_indices):
            leaf_points = plant_points[(plant_leaf_index == leaf_index) & (plant_is_focal == 1)]

            # if there are no points with this leaf index in this plant skip

            if leaf_points.shape[0] < down_sample_n:
                print(f"Insufficient points in leaf {leaf_points.shape[0]}")
                continue

            # otherwise compute the centroid
            leaf_centroid = np.median(leaf_points, axis=0)
            leaf_centroids.append(leaf_centroid)
            leaf_dataset['points'].append(leaf_points)
            leaf_dataset['leaf_index'].append(np.full(leaf_points.shape[0], leaf_index))
            leaf_dataset['leaf_count'].append(1)
            leaf_dataset['_plant_index'].append(plant_index[0]) #TODO check this

            leaf_ref_indices.append(leaf_total_count)
            leaf_total_count += 1
            leaf_count += 1

        print(f"[Info]     found {leaf_count} leaves")

        # plant_dataset['points'].append(plant_points);
        # plant_dataset['leaf_index'].append(plant_leaf_index)
        plant_dataset_additional['leaf_ref_index'].append(leaf_ref_indices)
        plant_dataset_additional['max_leaves'].append(leaf_count)
        plant_dataset_additional['leaf_centroids'].append(leaf_centroids)
        
    # We now have all single leaf data organized and can then generate some double
    # leaf data. We begin by calculating the centroid distances for each leaf pair
    # in each plant then we can take the k closest pairs.

    max_leaves_in_any_plant = np.max(np.array(plant_dataset_additional['max_leaves']))

    leaf_centroid_distances = np.full(
        (plant_count, max_leaves_in_any_plant, max_leaves_in_any_plant, 1), np.inf)
    leaf_point_lookup = np.zeros((plant_count, max_leaves_in_any_plant)) 

    print(f"[Info] Calculating centroids")

    for plant_idx in range(plant_count):
        centroids = plant_dataset_additional["leaf_centroids"][plant_idx]
        lookup = plant_dataset_additional["leaf_ref_index"][plant_idx]
        for i in range(len(centroids)):
            leaf_point_lookup[plant_idx, i] = lookup[i]
            for j in range(len(centroids)):
                if i == j:
                    continue

                cent_a = centroids[i]
                cent_b = centroids[j]

                dist = np.linalg.norm(cent_a - cent_b)

                leaf_centroid_distances[plant_idx, i, j, 0] = dist

    print(f"[Info] Matching leaves by proximity")

    # with the centroid distances calculated we can get the closest k pairs
    k = np.min(double_leaf_sample_n) # TODO be a bit smarter about this

    # 
    near_samples = np.argsort(leaf_centroid_distances, axis=None)
    idx = np.array(np.unravel_index(near_samples[0:k], leaf_centroid_distances.shape))

    single_sample_count = len(leaf_dataset["points"])

    if total_sample_n - double_leaf_sample_n > single_sample_count:
        print("[ERROR] Insufficient distinct single leaf samples.") 

    single_sample_idxs = np.random.choice(single_sample_count, total_sample_n - double_leaf_sample_n)
    paired_sample_idxs = np.array(list(range(single_sample_count, single_sample_count + k )))
    final_sample_idxs = np.concatenate((single_sample_idxs, paired_sample_idxs)) 

    for i in range(k):
        if leaf_centroid_distances[np.unravel_index(near_samples[i], leaf_centroid_distances.shape)] > centroid_dist_threshold:
            print(f"[ERROR] Insufficient closely paired samples in provided dataset, need {k}, found {i}.")
            break;
        centroid_idx = idx[:,i]
        plant_idx = centroid_idx[0] 
        leaf_a_idx = centroid_idx[1]
        leaf_b_idx = centroid_idx[2]
        idx_a = plant_dataset_additional["leaf_ref_index"][plant_idx][leaf_a_idx]
        idx_b = plant_dataset_additional["leaf_ref_index"][plant_idx][leaf_b_idx]

        add_merged(leaf_dataset, idx_a, idx_b)

    filter_samples(leaf_dataset, final_sample_idxs)

    # TODO we could do data augmentation here using rotations and skews if needed.

    # The k paired samples have now been added to the leaf_dataset we can now
    # perform any optional post-processing.
    
    # First we want to simulate segmentation errors in the leaf data. we will do
    # so by choosing a some point on the leaf or paired leaf and then choosing
    # only the nearest n points where n is governed by `segment_error_rate`
    # param. 
    
    if segment_error is not None:

        for i in range(len(leaf_dataset["points"])):

            if np.random.random() < segment_error_pop_percent:
                apply_segment_error(leaf_dataset, i, segment_error_rate)

    # We can then downsample so that each leaf has the same number of example
    # points we will eliminate any leafs that do not have sufficient points
    # available and warn.
    
    if down_sample_n is not None:
        for i in range(len(leaf_dataset["points"])):
            apply_down_sample(leaf_dataset, i, down_sample_n)

    return leaf_dataset


def split_dataset(dataset, split_ratio={'tr': 0.6, 'va': 0.2, 'te':0.2}):
    """
    Split a leaf_dataset into test, train, validation
    """
    dataset['points'] = np.stack(dataset['points'],0)
    dataset['leaf_index'] = np.array(dataset['leaf_index'])
    dataset['leaf_count'] = dataset['leaf_count']

    train_dataset = {k:[] for k in dataset.keys()}
    validation_dataset = {k:[] for k in dataset.keys()}
    test_dataset = {k:[] for k in dataset.keys()}

    train_dataset['points'], test_dataset['points'], \
    train_dataset['leaf_index'], test_dataset['leaf_index'], \
    train_dataset['leaf_count'], test_dataset['leaf_count'], \
    train_dataset['_plant_index'], test_dataset['_plant_index'], \
        = train_test_split(
            dataset['points'], 
            dataset['leaf_index'], 
            dataset['leaf_count'], 
            dataset['_plant_index'],
            test_size=split_ratio['te'], random_state=42)

    train_dataset['points'], validation_dataset['points'], \
    train_dataset['leaf_index'], validation_dataset['leaf_index'], \
    train_dataset['leaf_count'], validation_dataset['leaf_count'], \
    train_dataset['_plant_index'], validation_dataset['_plant_index'], \
        = train_test_split(
            train_dataset['points'], 
            train_dataset['leaf_index'], 
            train_dataset['leaf_count'], 
            train_dataset['_plant_index'],
            test_size=split_ratio['va']/(1-split_ratio['te']), random_state=42)   

    for k in train_dataset.keys():
        if k == 'names':
            continue
        train_dataset[k] = np.array(train_dataset[k])

    for k in validation_dataset.keys():
        if k == 'names':
            continue
        validation_dataset[k] = np.array(validation_dataset[k])
        
    for k in test_dataset.keys():
        if k == 'names':
            continue
        test_dataset[k] = np.array(test_dataset[k])

    return train_dataset, validation_dataset, test_dataset


def plot_leaf_dataset(leaf_dataset):
    """
    Util method for plotting full dataset, note use only for small test datasets
    """

    total_leafs = len(leaf_dataset["points"])
    if total_leafs > 300:
        print("Exceeded maximum supported data set size")

    grid_size = .4 
    grid_count = math.ceil(np.sqrt(min(total_leafs, 300)))

    plot = k3d.plot(name='points')

    idx = 0
    for i in range(grid_count):
        for j in range(grid_count):
            ridx = np.random.randint(0, total_leafs)

            if idx >= total_leafs:
                break
            points = leaf_dataset["points"][ridx]

            mean_adj = np.mean(points, axis=0)

            new_points = points + np.array([i * grid_size, j * grid_size, 0]) - mean_adj
            color = 0x0000ff if leaf_dataset["leaf_count"][ridx] == 1 else 0xff00ff 
            print(f'color {color} {leaf_dataset["leaf_count"][ridx]}')
            plt_points = k3d.points(positions=new_points, point_size=0.01, color=color)
            plot += plt_points
            idx += 1

    plt_points.shader='3d'
    plot.display()


def output_to_h5py(dataset, output_path):
    with h5py.File(output_path, 'w') as f:
        for k in dataset.keys():
            f.create_dataset(k, data=dataset[k])

def get_args():
    parser = argparse.ArgumentParser(
        description='Single and paired leaf dataset generation script.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-d',
                        '--input',
                        help='Path to input h5py processed training dataset.',
                        required=True)

    parser.add_argument('-o',
                        '--output',
                        help='Path to output directory.',
                        required=True)

    parser.add_argument('-c',
                        '--count',
                        help='Count of points in each sample',
                        required=False,
                        type=int,
                        default=80)

    parser.add_argument('-r',
                        '--threshold',
                        help='Maximum centroid distance allowed for close leaf pairs.',
                        metavar='threshold',
                        required=False,
                        type=float,
                        default=0.1)

    parser.add_argument('-t',
                        '--total',
                        help='Total number of samples to generate.',
                        required=False,
                        type=int,
                        default=200)

    return parser.parse_args()


def main():
    args = get_args()

    print(f"[Info] Looking for data at {args.input}")
    print(f"[Info] found:")

    print("[Info] Starting processing...")
    leaf_dataset = generate_leaf_dataset(args.input, down_sample_n=args.count, centroid_dist_threshold=args.threshold, total_sample_n=args.total)

    print(f"[Info] Successfully generated {len(leaf_dataset['points'])} leaves")
    (train_dataset, validation_dataset, test_dataset) = split_dataset(leaf_dataset)

    print(f"[Info] Saving results to {args.output}")

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    output_to_h5py(train_dataset, os.path.join(args.output, 'sorghum_leaf__labeled_train.hdf5'))
    output_to_h5py(validation_dataset, os.path.join(args.output, 'sorghum_leaf__labeled_validation.hdf5'))
    output_to_h5py(test_dataset, os.path.join(args.output, 'sorghum_leaf__labeled_test.hdf5'))

if __name__ == "__main__":
    main()