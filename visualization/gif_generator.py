import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import open3d as o3d
import random
import os
import json
import numpy as np


class PointCloudVisualizer:
    def __R_x(self, theta):
        return np.array(
            [
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)],
            ]
        )

    def __R_y(self, theta):
        return np.array(
            [
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)],
            ]
        )

    def __R_z(self, theta):
        return np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )

    def save_visualization(self, points, labels, file_path):
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        plt.grid(False)

        ims = []
        for rot in [self.__R_z]:
            for theta in np.linspace(0, 2 * np.pi, 40):
                rp = rot(theta).dot(points.T).T
                ims.append([ax.scatter(rp[:, 0], rp[:, 1], rp[:, 2], c=labels, s=1)])

        ani = animation.ArtistAnimation(fig, ims, blit=True)
        ani.save(file_path, writer="pillow", fps=7)
        plt.close(fig)


def normalize(points):
    mean = points.mean(axis=0)
    std = points.std(axis=0)
    points = (points - mean) / std
    return points


def labels_to_soil_and_lettuce_colors(labels):
    colors = np.array(["#4E342E" for i in range(labels.shape[0])])
    colors[labels == 1] = "#2E7D32"
    return colors


def save_gif(pcd_filename, output_filename):
    pcd = o3d.io.read_point_cloud(pcd_filename)
    R = pcd.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))
    pcd = pcd.rotate(R, center=pcd.get_center())
    points = np.array(pcd.points)
    labels = np.array(pcd.colors)
    vis = PointCloudVisualizer()
    points = normalize(points)
    vis.save_visualization(points, labels, output_filename)


def generate_bulk_gif(input_directory):
    folders = os.listdir(input_directory)
    for folder in folders:
        input_path = os.path.join(input_directory, folder, "semantic.ply")
        output_path = os.path.join(input_directory, folder, f"{folder}_semantic.gif")
        save_gif(input_path, output_path)
        print(f":: Generated gif for {folder}")


# generate_bulk_gif("/space/ariyanzarei/sorghum_segmentation/results/2020-08-06")
