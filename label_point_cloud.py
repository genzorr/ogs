import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial.distance import cdist
from tqdm import tqdm

from utils import (
    DEFAULT_COLOR,
    apply_color_palette,
    generate_color_palette,
    quat_trans_to_matrix,
    read_extrinsics,
    read_intrinsic,
)


def depth_to_point_cloud(
    rgb: np.ndarray,
    depth: np.ndarray,
    extrinsic: np.ndarray,
    segmentation_mask: np.ndarray,
):
    h, w = rgb.shape[:2]
    assert depth.shape[:2] == (h, w)

    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u = u.flatten()
    v = v.flatten()
    depth_map_flat = depth.flatten()

    x = (u - intrinsic[0, 2]) * depth_map_flat / intrinsic[0, 0]
    y = (v - intrinsic[1, 2]) * depth_map_flat / intrinsic[1, 1]
    z = depth_map_flat
    points = np.vstack((x, y, z)).T

    ones = np.ones((points.shape[0], 1))
    points_homogeneous = np.hstack((points, ones))
    points_world = points_homogeneous @ np.linalg.inv(extrinsic).T
    points_world = points_world[:, :3]

    colors = rgb[v, u] / 255.0

    centroids = []
    object_ids = np.zeros(points_world.shape[0], dtype=np.int32)
    for obj_idx in range(segmentation_mask.shape[0]):
        mask = segmentation_mask[obj_idx].flatten().astype(bool)
        object_ids[mask] = obj_idx + 1
        centroid = points_world[mask].mean(axis=0)
        centroids.append(centroid)

    return points_world, colors, object_ids, np.array(centroids)


def associate_objects(centroids_list):
    n_views = len(centroids_list)
    labels = [np.array([])] * n_views
    label_count = 0
    used_views = []

    max_view_idx = max((i for i in range(n_views) if i not in used_views), key=lambda i: len(centroids_list[i]))

    reference_centroids = centroids_list[max_view_idx]
    labels[max_view_idx] = np.arange(label_count, label_count + len(reference_centroids))
    label_count += len(reference_centroids)
    used_views.append(max_view_idx)

    for i in range(n_views):
        if i == max_view_idx or i in used_views:
            continue

        current_centroids = centroids_list[i]
        distances = cdist(current_centroids, reference_centroids)
        nearest_indices = np.argmin(distances, axis=1)

        current_labels = -np.ones(len(current_centroids), dtype=int)
        for k, nearest_idx in enumerate(nearest_indices):
            if np.min(distances[k]) < np.inf:
                current_labels[k] = labels[max_view_idx][nearest_idx]
            else:
                current_labels[k] = label_count
                label_count += 1

        labels[i] = current_labels
    return labels


def update_labels(labels: np.ndarray, label_association: np.ndarray):
    assert labels.max() == label_association.shape[0]

    new_labels = np.zeros_like(labels)
    for i, associated_label in enumerate(label_association):
        new_labels[labels == i + 1] = associated_label + 1

    return new_labels


def combine_point_clouds(point_clouds, colors):
    combined_points = np.vstack(point_clouds)
    combined_colors = np.vstack(colors)
    return combined_points, combined_colors


def create_point_cloud(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


SCENE = "test"  # replace with actual name
PROJECT_PATH = ""  # replace with actual path to root of the project

DATA_DIR = os.path.join(PROJECT_PATH, "gs2mesh", "data", "custom", SCENE)
OUTPUT_DIR = os.path.join(PROJECT_PATH, "gs2mesh", "output")

intrinsic_path = os.path.join(DATA_DIR, "sparse", "0", "cameras.txt")
extrinsics_path = os.path.join(DATA_DIR, "sparse", "0", "images.txt")
rgb_dir = os.path.join(DATA_DIR, "images")
depth_dir = os.path.join(OUTPUT_DIR, "custom_nw_iterations30000_DLNR_Middlebury_baseline7_0p/kitchen")
segmentation_mask_dir = os.path.join(OUTPUT_DIR, "masks")

w, h, fx, fy, cx, cy = read_intrinsic(intrinsic_path)
intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

selected_images = [1, 39, 62]
main_view_index = 2
far_plane = 8.0

extrinsics = read_extrinsics(extrinsics_path, selected_images)
extrinsics = [quat_trans_to_matrix(extrinsics[i]["q"], extrinsics[i]["t"]) for i in selected_images]

rgb_images, depth_maps, segmentation_masks = [], [], []

for i, image_id in enumerate(selected_images):
    rgb_path = os.path.join(rgb_dir, f"IMG_{((image_id - 1) * 10):05d}.png")
    depth_path = os.path.join(depth_dir, f"{(image_id - 1):03d}", "out_DLNR_Middlebury", "depth.npy")
    occlusion_mask_path = os.path.join(depth_dir, f"{(image_id - 1):03d}", "out_DLNR_Middlebury", "occlusion_mask.npy")
    segmentation_mask_path = os.path.join(segmentation_mask_dir, f"{(image_id - 1):03d}.npy")

    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    depth = np.load(depth_path)
    occlusion_mask = np.load(occlusion_mask_path)
    segmentation_mask = np.load(segmentation_mask_path)

    # Apply occlusion mask from g2mesh to remove outliers,
    # which are seen only on one of cameras from stereo view
    depth = depth * occlusion_mask
    segmentation_mask = segmentation_mask * occlusion_mask

    # Apply far plane to speed up computations
    depth[depth > far_plane] = 0.0

    rgb_images.append(rgb)
    depth_maps.append(depth)
    segmentation_masks.append(segmentation_mask)

# Create color palette
K = np.max([mask.shape[0] for mask in segmentation_masks])
color_palette = generate_color_palette(K, cmap_name="tab10")
print(f"Number of objects: {K}")

point_clouds = []
object_labels = []
colors = []
centroids_list = []

# Process all views
for i in tqdm(range(len(selected_images))):
    points, clr, labels, centroids = depth_to_point_cloud(
        rgb_images[i], depth_maps[i], extrinsics[i], segmentation_masks[i]
    )

    point_clouds.append(points)
    object_labels.append(labels)
    colors.append(clr)
    centroids_list.append(centroids)

# Associate objects across views
label_association = associate_objects(centroids_list)

total_labels = []
# Update the labels in the point clouds
for i in range(len(point_clouds)):
    updated_labels = update_labels(object_labels[i], label_association[i])
    apply_color_palette(updated_labels, colors[i], color_palette)
    total_labels.append(updated_labels)

# Export points corresponding to objects in the combined point cloud
object_points = {i: np.array([], dtype=np.float32).reshape(0, 3) for i in range(K + 1)}
for i, labels in enumerate(total_labels):
    zero_mask = np.ones_like(total_labels[0], dtype=bool)
    for obj_id in range(1, K + 1):
        mask = total_labels[i] == obj_id
        object_points[obj_id] = np.vstack((object_points[obj_id], point_clouds[i][mask]))

        zero_mask &= ~mask

    # Add other gaussian means (non-object)
    mask = zero_mask
    object_points[0] = np.vstack((object_points[0], point_clouds[i][mask]))

# Merge all object points
points = np.array([], dtype=np.float32).reshape(0, 3)
point_labels = np.array([], dtype=np.int32).reshape(0, 1)
for obj_id, obj_pts in object_points.items():
    points = np.vstack((points, obj_pts))
    point_labels = np.vstack((point_labels, np.full((obj_pts.shape[0], 1), obj_id)))

np.savez(os.path.join(OUTPUT_DIR, "object_points.npz"), points=points, labels=point_labels)

# Export merged object points to Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector([color_palette.get(obj_id, DEFAULT_COLOR) for obj_id in point_labels.flatten()])
o3d.visualization.draw_geometries([pcd])
