import matplotlib.pyplot as plt
import numpy as np
import transforms3d

DEFAULT_COLOR = [0.0, 0.0, 0.0]


def read_intrinsic(path):
    with open(path, "r") as f:
        for _ in range(4):
            line = f.readline()

        line = line.strip().split()
        w, h = [int(x) for x in line[2:4]]
        fx, fy, cx, cy = [float(x) for x in line[4:]]
    return w, h, fx, fy, cx, cy


def read_extrinsics(path, selected_images):
    selected_images_str = [str(i) for i in selected_images]
    extrinsics = {}

    with open(path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break

            line = line.strip().split()

            if line[0] in selected_images_str:
                image_number = int(line[0])
                q = np.array([float(x) for x in line[1:5]])
                t = np.array([float(x) for x in line[5:8]])

                extrinsics[image_number] = {"q": q, "t": t}
    return extrinsics


def quat_trans_to_matrix(quat, trans):
    R = transforms3d.quaternions.quat2mat(quat)

    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = trans
    return extrinsic


def generate_color_palette(num_colors, cmap_name="tab10"):
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(0, 1, num_colors))
    color_palette = {i + 1: colors[i, :3] for i in range(num_colors)}
    return color_palette


def apply_color_palette(object_ids, colors, color_palette):
    for obj_id in range(1, int(np.max(object_ids)) + 1):
        mask = object_ids == obj_id
        colors[mask] = color_palette.get(obj_id, DEFAULT_COLOR)
