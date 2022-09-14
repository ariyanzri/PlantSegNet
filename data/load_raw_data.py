import open3d as o3d
import numpy as np
import random
from plyfile import PlyData, PlyElement
from data.utils import *


def load_pcd_plyfile(path, index_to_use="leaf_index", down_sample_n=8000):
    try:
        with open(path, "rb") as f:
            plydata = PlyData.read(f)
            points = np.asarray(np.array(plydata.elements[0].data).tolist())
            points_full = np.asarray(np.array(plydata.elements[0].data).tolist())

            if down_sample_n is None:
                down_sample_n = points_full.shape[0]

            downsample_indexes = random.sample(
                np.arange(0, points.shape[0]).tolist(),
                min(down_sample_n, points.shape[0]),
            )
            points = points[downsample_indexes]

            leaf_index = np.asarray(np.array(plydata.elements[2].data).tolist())[
                downsample_indexes
            ].squeeze()
            leaf_part_index = np.asarray(np.array(plydata.elements[3].data).tolist())[
                downsample_indexes
            ].squeeze()

            if np.min(leaf_part_index) == 1:
                leaf_part_index -= 1

            is_focal_plant = np.asarray(np.array(plydata.elements[4].data).tolist())[
                downsample_indexes
            ].squeeze()

            if len(plydata.elements) == 6:
                plant_index = np.asarray(np.array(plydata.elements[5].data).tolist())[
                    downsample_indexes
                ].squeeze()
            elif len(plydata.elements) == 7:
                plant_index = np.asarray(np.array(plydata.elements[5].data).tolist())[
                    downsample_indexes
                ].squeeze()
                ground_index = np.asarray(np.array(plydata.elements[6].data).tolist())[
                    downsample_indexes
                ].squeeze()
            else:
                plant_index = None
                ground_index = None

            leaf_part_full_index = np.zeros(leaf_index.shape)

            min_leaf_index = np.min(leaf_index)
            max_leaf_index = np.max(leaf_index)
            min_leaf_part_index = np.min(leaf_part_index)
            max_leaf_part_index = np.max(leaf_part_index)

            k = 0
            for i in range(min_leaf_index, max_leaf_index + 1):
                for j in range(min_leaf_part_index, max_leaf_part_index + 1):
                    if not np.any((leaf_index == i) & (leaf_part_index == j)):
                        continue
                    leaf_part_full_index[(leaf_index == i) & (leaf_part_index == j)] = k
                    k += 1

        return {
            "points_full": points_full,
            "points": points,
            "is_focal_plant": is_focal_plant,
            "leaf_index": leaf_index,
            "leaf_part_index": leaf_part_index,
            "leaf_part_full_index": leaf_part_full_index,
            "plant_index": plant_index,
            "ground_index": ground_index,
        }

    except Exception as e:
        print(e)
        return None


def load_ply_file_points(path, n_points=8000):
    pcd = o3d.io.read_point_cloud(path)

    R = pcd.get_rotation_matrix_from_xyz((-np.pi / 2, 0, 0))
    pcd = pcd.rotate(R, center=pcd.get_center())

    points = np.array(pcd.points)
    # points[:, 0], points[:, 1], points[:, 2] = points[:, 2], points[:, 0], points[:, 1]

    first_down_indexes = random.sample(
        np.arange(0, points.shape[0]).tolist(),
        min(50000, points.shape[0]),
    )
    points = points[first_down_indexes]

    downsample_indexes = random.sample(
        np.arange(0, points.shape[0]).tolist(),
        min(n_points, points.shape[0]),
    )
    down_sampled_points = points[downsample_indexes]

    return points, down_sampled_points


def paint_pcd_into_o3d(plyfile_pcd, key):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(plyfile_pcd["points"])
    pcd.paint_uniform_color([0, 0, 0])
    colors = np.array(pcd.colors)
    ind_min = np.min(plyfile_pcd[key])
    ind_max = np.max(plyfile_pcd[key])

    d_colors = distinct_colors()

    # for i in range(ind_min,ind_max+1):
    #    colors[plyfile_pcd[key][:,0]==i,:] = d_colors[i+1]

    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd
