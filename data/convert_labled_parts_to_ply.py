import open3d as o3d
import numpy as np
import os
import pandas as pd
from plyfile import PlyData, PlyElement


def merge_labled_parts(path, outpath):
    full_points = []
    full_leaf_indices = []
    full_semantic_indices = []

    files = os.listdir(path)
    for file in files:
        pcd = o3d.io.read_point_cloud(os.path.join(path, file))
        points = np.array(pcd.points).astype("float64")

        if "leaf" in file:
            leaves = np.full(
                points.shape[0], int(file.replace("leaf_", "").replace(".pts", ""))
            )
            semantics = np.ones(points.shape[0])
        elif "ground" in file:
            leaves = np.full(points.shape[0], -1)
            semantics = np.zeros(points.shape[0])
        elif "non_focal" in file:
            leaves = np.full(points.shape[0], -1)
            semantics = np.full(points.shape[0], 2)
        else:
            continue

        full_points.append(points)
        full_leaf_indices.append(leaves)
        full_semantic_indices.append(semantics)

    full_points = np.concatenate(full_points, axis=0)
    full_leaf_indices = np.concatenate(full_leaf_indices, axis=0)
    full_semantic_indices = np.concatenate(full_semantic_indices, axis=0)

    x, y, z = full_points[:, 0], full_points[:, 1], full_points[:, 2]
    full_points = list(zip(x, y, z))

    points_array = np.array(full_points, dtype=[("x", "f8"), ("y", "f8"), ("z", "f8")])

    leaf_indices_array = np.array(full_leaf_indices, dtype=[("leaf_indices", "i4")])

    semantic_indices_array = np.array(
        full_semantic_indices, dtype=[("semantic_indices", "i4")]
    )

    points_element = PlyElement.describe(points_array, "vertex")
    leaf_element = PlyElement.describe(leaf_indices_array, "leaf_index")
    semantic_element = PlyElement.describe(semantic_indices_array, "semantic_index")

    PlyData(
        [
            points_element,
            leaf_element,
            semantic_element,
        ]
    ).write(outpath)

    # points = pd.DataFrame(
    #     {
    #         "x": full_points[:, 0],
    #         "y": full_points[:, 1],
    #         "z": full_points[:, 2],
    #         "leaf_index": full_leaf_indices,
    #         "semantic_index": full_semantic_indices,
    #     }
    # )
    # cloud = PyntCloud(points)
    # cloud.to_file(outpath)


merge_labled_parts(
    "/space/ariyanzarei/sorghum_segmentation/dataset/real_data/labeled/raw/BTx_623_3001_330755982981",
    "/space/ariyanzarei/sorghum_segmentation/dataset/real_data/labeled/ply_files/BTx_623_3001_330755982981.ply",
)
