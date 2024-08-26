import os
from typing import Dict

import numpy as np
import open3d as o3d
import torch
from scipy.spatial import cKDTree

from utils import generate_color_palette


def gs_assign_labels(means: np.ndarray, object_points_dict: Dict[int, np.ndarray], labels: np.ndarray):
    # Build a KDTree for the Gaussian means
    means_kdtree = cKDTree(means)

    for object_id, object_points in object_points_dict.items():
        _, idx = means_kdtree.query(object_points)
        labels[idx] = object_id


SCENE = "test"  # replace with actual name
PROJECT_PATH = ""  # replace with actual path to root of the project
DATA_DIR = os.path.join(PROJECT_PATH, "gs2mesh", "data", "custom", SCENE)

ckpt_path = os.path.join(DATA_DIR, "results", "ckpts", "ckpt_29999_rank0.pt")
object_points_path = os.path.join(PROJECT_PATH, "gs2mesh", "output", "object_points.npz")

ckpt = torch.load(ckpt_path)["splats"]
means = ckpt["means"].cpu().numpy()

# Remove transform, applied by gsplat (normalization).
# This can be extracted from gsplat's colmap parser, or you can disable normalization in gsplat.
transform = np.array(
    [
        [0.06874676, -0.15988725, 0.16953212, 0.06171692],
        [-0.23127033, -0.02510266, 0.07010761, 0.12246974],
        [-0.02861998, -0.18121013, -0.15929523, 0.71876923],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
transform = np.linalg.inv(transform)
means = means @ transform[:3, :3].T + transform[:3, 3]

object_points = np.load(object_points_path, allow_pickle=True)
points = object_points["points"]
point_labels = object_points["labels"]

K = np.max(point_labels)
color_palette = generate_color_palette(K)

points_kdtree = cKDTree(points)
_, indices = points_kdtree.query(means)
gs_labels = point_labels[indices].reshape(-1)

points = np.array([], dtype=np.float32).reshape(0, 3)
colors = np.array([], dtype=np.float32).reshape(0, 3)

for object_id in range(1, K + 1):
    object_means = means[gs_labels == object_id]
    points = np.vstack((points, object_means))
    colors = np.vstack((colors, np.tile(color_palette[object_id], (object_means.shape[0], 1))))

object_means = means[gs_labels == 0]
points = np.vstack((points, object_means))
colors = np.vstack((colors, np.tile([0.5, 0.5, 0.5], (object_means.shape[0], 1))))

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([pcd])

# Downsample to remove duplicates and redundant points
voxel_size = 0.01
pcd = pcd.voxel_down_sample(voxel_size)
o3d.io.write_point_cloud(os.path.join(PROJECT_PATH, "g2mesh", "output", "labeled_gs_means.ply"), pcd)
