import torch
import numpy as np
import torch.utils.data as data
import h5py
import random
import os


class SorghumDataset(data.Dataset):
    """'
    Semantic label guide:
        * 0 --> ground
        * 1 --> focal plant
        * 2 --> surrounding plants

    Ground label guide:
        * 0 --> not ground
        * 1 --> ground

    """

    def __init__(self, h5_filename):
        super().__init__()
        self.h5_filename = h5_filename
        self.length = -1

    def __getitem__(self, index):
        f = h5py.File(self.h5_filename, "r")
        points = f["points"][index]
        is_focal_plant = f["is_focal_plant"][index]
        ground_index = f["ground_index"][index]
        plant_index = f["plant_index"][index]
        leaf_index = f["leaf_index"][index]

        # Converting arbitrary and non-contigiouse plant IDs to contigiouse list of indices
        plant_ind = list(set(list(plant_index)))
        ind = list(range(0, len(plant_ind)))
        mapping = dict(zip(ind, plant_ind))
        new_plant = np.zeros(plant_index.shape)
        for key in mapping:
            new_plant[plant_index == mapping[key]] = key
        plant_index = new_plant

        # creating semantic labeling using the guide above
        semantic_label = is_focal_plant.copy()
        semantic_label[np.where((is_focal_plant == 0) & (ground_index == 1))] = 0
        semantic_label[np.where((is_focal_plant == 0) & (ground_index == 0))] = 2

        f.close()

        return (
            torch.from_numpy(points).float(),
            torch.from_numpy(ground_index).float(),
            torch.from_numpy(semantic_label).type(torch.LongTensor),
            torch.from_numpy(plant_index).type(torch.LongTensor),
            torch.from_numpy(leaf_index).type(torch.LongTensor),
        )

    def __len__(self):
        if self.length != -1:
            return self.length
        else:
            f = h5py.File(self.h5_filename, "r")
            self.length = len(f["names"])
            f.close()
            return self.length

    def get_name(self, index):
        f = h5py.File(self.h5_filename, "r")
        name = f["names"][index].decode("utf-8")
        return name


class SorghumDatasetWithNormals(data.Dataset):
    """'
    Semantic label guide:
        * 0 --> ground
        * 1 --> focal plant
        * 2 --> surrounding plants

    """

    def __init__(
        self,
        h5_filename,
        use_normals=True,
        std_coef=0.015,
        duplicate_p=0,
        focal_only_p=0,
        debug=False,
    ):
        super().__init__()
        self.h5_filename = h5_filename
        self.length = -1
        self.append_normals = use_normals
        self.std_coef = std_coef
        self.is_semantic = "semantic" in h5_filename
        self.duplicate_ground_probability = duplicate_p
        self.focal_only_probability = focal_only_p
        self.is_debug = debug

    def semantic_transformations(self, np_points, np_labels):

        dimension_sizes = np.max(np_points, 0) - np.min(np_points, 0)

        # ground transformations

        rnd_num = np.random.rand(1)
        if rnd_num < self.duplicate_ground_probability:
            # add duplicate to the ground points (with probability)
            ground = np.where(np_labels == 0)[0]
            duplicate_indices = np.random.choice(
                ground.tolist(),
                int(ground.shape[0] / 2),
                replace=False,
            )
            np_points[duplicate_indices, 2] -= dimension_sizes[2] / 10

        initial_size = np_points.shape[0]

        rnd_num = np.random.rand(1)
        if rnd_num < self.focal_only_probability:
            # keep only focal plant
            np_points = np_points[(np_labels == 1) | (np_labels == 0), :]
            np_labels = np_labels[(np_labels == 1) | (np_labels == 0)]

            focal_indices = np.random.choice(
                np.arange(0, np_points.shape[0]).tolist(),
                initial_size,
                replace=True,
            )
            np_points = np_points[focal_indices]
            np_labels = np_labels[focal_indices]

        return np_points, np_labels

    def __getitem__(self, index):
        f = h5py.File(self.h5_filename, "r")
        np_points = f["points"][index]
        np_labels = f["labels"][index].squeeze()
        np_normals = f["normals"][index]

        # perform data augmentation if semantic segmentation
        if self.is_semantic:
            np_points, np_labels = self.semantic_transformations(np_points, np_labels)

        # add noise to all points
        if self.std_coef > 0:
            std_points = np.repeat(
                np.expand_dims(np.std(np_points, 0), 0), np_points.shape[0], 0
            )
            np_points += np.random.normal(
                0, std_points * self.std_coef, size=np_points.shape
            )

        # convert to torch
        points = torch.from_numpy(np_points).type(torch.DoubleTensor)
        labels = torch.from_numpy(np_labels).type(torch.LongTensor)
        normals = torch.from_numpy(np_normals).float()
        f.close()

        if self.append_normals:
            features = torch.cat((points, normals), -1)
        else:
            features = points

        return (
            features,
            labels,
        )

    def __len__(self):
        if self.length != -1:
            return self.length
        else:
            f = h5py.File(self.h5_filename, "r")
            self.length = len(f["points"])
            f.close()
            if self.is_debug:
                self.length = 50
            return self.length


class LeafDataset(data.Dataset):
    """'
    Leaf label guide:
        * 0 --> single leaf
        * 1 --> double leaf
    """

    def __init__(self, h5_filename):
        super().__init__()
        self.h5_filename = h5_filename
        self.length = -1

    def __getitem__(self, index):
        f = h5py.File(self.h5_filename, "r")
        points = f["points"][index]

        # Note leaf counts are expected to be either 1 or 2
        is_single_leaf = 2 - f["leaf_count"][index]

        f.close()

        return (
            torch.from_numpy(points).float(),
            torch.from_numpy(np.array(is_single_leaf)).type(torch.LongTensor),
        )

    def __len__(self):
        if self.length != -1:
            return self.length
        else:
            f = h5py.File(self.h5_filename, "r")
            self.length = len(f["points"])
            f.close()
            return self.length

    def get_name(self, index):
        f = h5py.File(self.h5_filename, "r")
        name = f["names"][index].decode("utf-8")
        return name


class PartNetDataset(data.Dataset):
    def __init__(self, path, debug=False):
        super().__init__()
        self.h5_filename = path
        self.is_debug = debug
        if not os.path.exists(self.h5_filename):
            raise Exception(
                "H5py file doesn't exist. Please make sure you are inputing correct values."
            )
        self.length = -1

    def __getitem__(self, index):
        f = h5py.File(self.h5_filename, "r")
        np_points = f["pts"][index]
        np_labels = f["label"][index].squeeze()

        # convert to torch
        points = torch.from_numpy(np_points).type(torch.DoubleTensor)
        labels = torch.from_numpy(np_labels).type(torch.LongTensor)
        f.close()

        return (
            points,
            labels,
        )

    def __len__(self):
        if self.length != -1:
            return self.length
        else:
            f = h5py.File(self.h5_filename, "r")
            self.length = len(f["pts"])
            f.close()
            if self.is_debug:
                self.length = 50
            return self.length


class TreePartNetDataset(data.Dataset):
    def __init__(self, path, debug=False):
        super().__init__()
        self.h5_filename = path
        self.is_debug = debug
        if not os.path.exists(self.h5_filename):
            raise Exception(
                "H5py file doesn't exist. Please make sure you are inputing correct values."
            )
        self.length = -1

    def __getitem__(self, index):
        f = h5py.File(self.h5_filename, "r")
        np_points = f["points"][index]
        np_labels = f["primitive_id"][index].squeeze()

        # convert to torch
        points = torch.from_numpy(np_points).type(torch.DoubleTensor)
        labels = torch.from_numpy(np_labels).type(torch.LongTensor)
        f.close()

        return (
            points,
            labels,
        )

    def __len__(self):
        if self.length != -1:
            return self.length
        else:
            f = h5py.File(self.h5_filename, "r")
            self.length = len(f["points"])
            f.close()
            if self.is_debug:
                self.length = 50
            return self.length
