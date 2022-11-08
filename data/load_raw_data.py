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


def load_pcd_plyfile_new_approach(path, is_instance, down_sample_n=8000):
    try:
        with open(path, "rb") as f:
            plydata = PlyData.read(f)
            points = np.asarray(np.array(plydata.elements[0].data).tolist())
            points_full = np.asarray(np.array(plydata.elements[0].data).tolist())
            leaf_index = np.asarray(np.array(plydata.elements[2].data).tolist())
            ground_index = np.asarray(np.array(plydata.elements[6].data).tolist())
            is_focal_plant = np.asarray(np.array(plydata.elements[4].data).tolist())

            if down_sample_n is None:
                down_sample_n = points_full.shape[0]

            if not is_instance:
                downsample_indexes = random.sample(
                    np.arange(0, points.shape[0]).tolist(),
                    min(down_sample_n, points.shape[0]),
                )
                points = points[downsample_indexes]
                is_focal_plant = is_focal_plant[downsample_indexes].squeeze()
                ground_index = ground_index[downsample_indexes].squeeze()

                label = np.zeros(is_focal_plant.shape)
                label[(is_focal_plant == 0) & (ground_index == 0)] = 2
                label[(is_focal_plant == 1) & (ground_index == 0)] = 1

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamRadius(0.05)
                )
                normals = np.asarray(pcd.normals)

            else:
                is_focal_plant = is_focal_plant.squeeze()
                ground_index = ground_index.squeeze()

                points = points[(is_focal_plant == 1) & (ground_index == 0), :]
                points_full = points_full[
                    (is_focal_plant == 1) & (ground_index == 0), :
                ]
                label = leaf_index[(is_focal_plant == 1) & (ground_index == 0)]

                points_shape = points.shape[0]

                downsample_indexes = np.random.choice(
                    np.arange(0, points.shape[0]).tolist(),
                    down_sample_n,
                    replace=(True if points_shape < down_sample_n else False),
                )

                points = points[downsample_indexes]
                label = label[downsample_indexes]

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamRadius(0.05)
                )
                normals = np.asarray(pcd.normals)

        return {
            "points_full": points_full,
            "points": points,
            "labels": label,
            "normals": normals,
        }

    except Exception as e:
        print(e)
        return None


def load_ply_file_points(path, n_points=8000, full_points=50000):
    pcd = o3d.io.read_point_cloud(path)

    # R = pcd.get_rotation_matrix_from_xyz((-np.pi / 2, 0, 0))
    # pcd = pcd.rotate(R, center=pcd.get_center())

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    normals = np.asarray(pcd.normals)

    points = np.array(pcd.points)
    # points[:, 0], points[:, 1], points[:, 2] = points[:, 2], points[:, 0], points[:, 1]

    first_down_indexes = random.sample(
        np.arange(0, points.shape[0]).tolist(),
        min(full_points, points.shape[0]),
    )
    points = points[first_down_indexes]

    downsample_indexes = random.sample(
        np.arange(0, points.shape[0]).tolist(),
        min(n_points, points.shape[0]),
    )
    down_sampled_points = points[downsample_indexes]
    normals = normals[downsample_indexes]

    return points, down_sampled_points, normals


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
