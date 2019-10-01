import os, sys
import numpy as np

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from PIL import Image, ImageDraw

# keypoints
keypoint_names = (
    'Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck',
    'Nose', 'Head', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist'
)

keypoint_count = len(keypoint_names)

# keypoint connection
connections = (
    {0, 7}, {7, 8}, {8, 9}, {9, 10}, {8, 11}, {11, 12}, {12, 13}, {8, 14},
    {14, 15}, {15, 16}, {0, 1}, {1, 2}, {2, 3}, {0, 4}, {4, 5}, {5, 6}
)

connection_count = len(connections)

# flip position-aware keypoints
connections_flipped = list()

for i, j in connections:
    if keypoint_names[i].startswith('L_') or keypoint_names[j].startswith('L_'):
        new_i = keypoint_names.index(keypoint_names[i].replace('L_', 'R_'))
        new_j = keypoint_names.index(keypoint_names[j].replace('L_', 'R_'))
    elif keypoint_names[i].startswith('R_') or keypoint_names[j].startswith('R_'):
        new_i = keypoint_names.index(keypoint_names[i].replace('R_', 'L_'))
        new_j = keypoint_names.index(keypoint_names[j].replace('R_', 'L_'))
    else:
        new_i, new_j = i, j

    connections_flipped.append({new_i, new_j})

# color of keypoints
keypoint_2d_color = 'gray'
keypoint_3d_color = 'white'

# colormap of the skeleton
skeleton_colormap_name = 'jet'
skeleton_colormap = cm.get_cmap(skeleton_colormap_name)
skeleton_colors = list()

for i in range(connection_count):
    key = float(i) / (connection_count - 1)  # RGBA in [0, 1] range
    skeleton_colors.append(skeleton_colormap(key))

depth_color_map = 'rainbow'

depth_inf = 2.0

# keypoint and skeleton size
keypoint_size = 10
skeleton_width = 5


def _depth_normalize_1(depth):
    normalized_depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    return normalized_depth


def _depth_normalize_2(depth, basepoint='Torso', normalize_factor=150.0):
    idx = keypoint_names.index(basepoint)
    basepoint_depth = depth[idx]

    normalized_depth = (depth - basepoint_depth) / normalize_factor
    return normalized_depth


def _interpolate_depth(x, y, x1, y1, z1, x2, y2, z2):
    d = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # project (x1, x) onto (x1, x2)
    d1 = (x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)
    d1 = np.abs(d1 / d)

    # project (x2, x) onto (x2, x1)
    d2 = (x - x2) * (x1 - x2) + (y - y2) * (y1 - y2)
    d2 = np.abs(d2 / d)

    z = (z1 * d2 + z2 * d1) / d
    return z


# draw skeleton
def _draw_skeleton_2d(img, c, keypoints_2d, normalized_depth):
    w, h = img.size

    # record the minimum depth of each point
    depth_buffer = np.full((h, w), depth_inf, dtype=np.float32)
    fused = np.array(img)

    for t, (i, j) in enumerate(c):
        x1, y1 = keypoints_2d[i]
        z1 = normalized_depth[i]
        x2, y2 = keypoints_2d[j]
        z2 = normalized_depth[j]

        r, g, b, _ = skeleton_colors[t]
        r = int(round(255 * r))
        g = int(round(255 * g))
        b = int(round(255 * b))
        color = np.array((r, g, b), dtype=np.uint8)

        img_line = Image.new('L', (w, h), 'black')  # black:0, white: 255
        draw = ImageDraw.ImageDraw(img_line)
        draw.line((x1, y1, x2, y2), fill='white', width=skeleton_width)

        nonzero_y, nonzero_x = np.nonzero(np.array(img_line))  # x: horizontal, v: vertical
        z = _interpolate_depth(nonzero_x, nonzero_y, x1, y1, z1, x2, y2, z2)

        # True: replace, False: leave alone
        mask = np.less(z, depth_buffer[nonzero_y, nonzero_x]).astype(np.float32)

        depth_buffer[nonzero_y, nonzero_x] = (1 - mask) * depth_buffer[nonzero_y, nonzero_x] + mask * z

        mask_expand = np.expand_dims(mask, -1)
        fused[nonzero_y, nonzero_x] = (1 - mask_expand) * fused[nonzero_y, nonzero_x] + mask_expand * color

    fused_img = Image.fromarray(fused)
    return fused_img


# draw circle keypoints
def _draw_keypoints_2d(img, keypoints_2d):
    draw = ImageDraw.ImageDraw(img)

    for i in range(keypoint_count):
        x, y = keypoints_2d[i]
        radius = float(keypoint_size) / 2
        bbox = (x - radius, y - radius, x + radius, y + radius)
        draw.ellipse(bbox, fill=keypoint_2d_color)

    return img


def skeleton_2d(image_size, keypoints_2d, depth, flip, _DEBUG=False, original_img=None):
    if flip:
        c = connections_flipped
    else:
        c = connections

    img = Image.new('RGB', image_size, color='black')

    normalized_depth = _depth_normalize_1(depth)  # [0, 1]

    img = _draw_skeleton_2d(img, c, keypoints_2d, normalized_depth)
    img = _draw_keypoints_2d(img, keypoints_2d)

    if _DEBUG:
        plt.clf()
        plt.imshow(img)  # skeleton
        plt.show()

        original_img2 = original_img.copy()
        original_img2 = _draw_skeleton_2d(original_img2, c, keypoints_2d, normalized_depth)
        original_img2 = _draw_keypoints_2d(original_img2, keypoints_2d)

        plt.clf()
        plt.imshow(original_img2)  # skeleton drawn over the original image
        plt.show()

    return img


def skeleton_rgbd(image_size, keypoints_2d, depth, flip, _DEBUG=False, original_img=None):
    if flip:
        c = connections_flipped
    else:
        c = connections

    w, h = image_size
    depth_img = np.full((h, w), depth_inf, dtype=np.float32)

    normalized_depth = _depth_normalize_1(depth)

    # draw skeleton
    for i, j in c:
        x1, y1 = keypoints_2d[i]
        z1 = normalized_depth[i]
        x2, y2 = keypoints_2d[j]
        z2 = normalized_depth[j]

        img_line = Image.new('L', image_size, 'black')  # black:0, white: 255
        draw = ImageDraw.ImageDraw(img_line)
        draw.line((x1, y1, x2, y2), fill='white', width=skeleton_width)

        nonzero_y, nonzero_x = np.nonzero(np.array(img_line))  # x: horizontal, v: vertical
        z = _interpolate_depth(nonzero_x, nonzero_y, x1, y1, z1, x2, y2, z2)

        depth_img[(nonzero_y, nonzero_x)] = np.minimum(depth_img[(nonzero_y, nonzero_x)], z)

    depth_img[depth_img == depth_inf] = -1.0  # -1.0, [0, 1]

    # draw circle keypoints
    for i in range(keypoint_count):
        x, y = keypoints_2d[i]
        z = normalized_depth[i]
        radius = float(keypoint_size) / 2
        bbox = (x - radius, y - radius, x + radius, y + radius)

        img_circle = Image.new('L', image_size, 'black')
        draw = ImageDraw.ImageDraw(img_circle)
        draw.ellipse(bbox, fill='white')
        nonzero_pos = np.nonzero(np.array(img_circle))

        depth_img[nonzero_pos] = z

    if _DEBUG:
        colormap = cm.get_cmap(depth_color_map)
        depth_img2 = colormap(depth_img)  # RGBA
        depth_img2[depth_img == -1.0] = np.array(colors.to_rgba('black'))
        depth_img2 = np.round(255.0 * depth_img2).astype(np.uint8)

        plt.clf()
        plt.imshow(Image.fromarray(depth_img2))  # depth image
        plt.show()

        original_img2 = np.array(original_img.convert('RGBA'))
        nonzero_pos = np.nonzero(depth_img + 1.0)
        original_img2[nonzero_pos] = depth_img2[nonzero_pos]

        plt.clf()
        plt.imshow(Image.fromarray(original_img2))  # depth image drawn over the original image
        plt.show()

    skeleton_img = skeleton_2d(image_size, keypoints_2d, depth, flip)
    return skeleton_img, depth_img


def skeleton_3d(image_size, depth_voxel, keypoints_2d, depth, flip, _DEBUG=False, original_img=None):
    if flip:
        c = connections_flipped
    else:
        c = connections

    w, h = image_size
    voxel_rgba = np.zeros((h, w, depth_voxel, 4), dtype=np.float32)

    normalized_depth = _depth_normalize_1(depth)

    # draw skeleton
    for t, (i, j) in enumerate(c):
        x1, y1 = keypoints_2d[i]
        z1 = normalized_depth[i]
        x2, y2 = keypoints_2d[j]
        z2 = normalized_depth[j]

        img_line = Image.new('L', image_size, 'black')  # black:0, white: 255
        draw = ImageDraw.ImageDraw(img_line)
        draw.line((x1, y1, x2, y2), fill='white', width=skeleton_width)

        nonzero_y, nonzero_x = np.nonzero(np.array(img_line))  # x: horizontal, v: vertical
        z = _interpolate_depth(nonzero_x, nonzero_y, x1, y1, z1, x2, y2, z2)

        nonzero_z = np.round(z * (depth_voxel - 1)).astype(np.int64)  # array
        voxel_rgba[nonzero_y, nonzero_x, nonzero_z, ...] = np.array(skeleton_colors[t])  # RGBA

    # draw circle keypoints
    for i in range(keypoint_count):
        x, y = keypoints_2d[i]
        z = normalized_depth[i]
        radius = float(keypoint_size) / 2
        bbox = (x - radius, y - radius, x + radius, y + radius)

        img_circle = Image.new('L', image_size, 'black')
        draw = ImageDraw.ImageDraw(img_circle)
        draw.ellipse(bbox, fill='white')
        nonzero_y, nonzero_x = np.nonzero(np.array(img_circle))

        nonzero_z = np.round(z * (depth_voxel - 1)).astype(np.int64)  # scalar
        voxel_rgba[nonzero_y, nonzero_x, nonzero_z, ...] = np.array(colors.to_rgba(keypoint_3d_color))

    voxel_rgb = voxel_rgba[:, :, :, :-1]  # (h, w, depth_voxel, 3)
    voxel_rgb = np.round(255.0 * voxel_rgb).astype(np.uint8)  # [0, 255]

    if _DEBUG:
        voxel_mask = np.sum(voxel_rgb, axis=-1).astype(np.bool_)  # (h, w, depth_voxel)

        # voxel_mask_transposed = np.transpose(voxel_mask, (1, 2, 0))  # (w, depth_voxel, h)
        # voxel_rgba_transposed = np.transpose(voxel_rgba, (1, 2, 0, 3))  # (w, depth_voxel, h, 3)
        #
        # plt.clf()
        # fig = plt.gcf()
        # ax = Axes3D(fig)
        # ax.voxels(filled=voxel_mask_transposed, facecolors=voxel_rgba_transposed, linewidth=0.0)  # 3d skeleton
        # plt.show()

        min_depth = np.argmax(voxel_mask, axis=-1)
        min_depth_flatten = np.reshape(min_depth, -1)
        voxel_rgb_flatten = np.reshape(voxel_rgb, (-1, depth_voxel, 3))
        projected_img_flatten = voxel_rgb_flatten[np.arange(h * w), min_depth_flatten, ...]
        projected_img = np.reshape(projected_img_flatten, (h, w, 3))

        original_img2 = np.array(original_img.convert('RGB'))
        nonzero_pos = np.nonzero(np.sum(projected_img, axis=-1))
        original_img2[nonzero_pos] = projected_img[nonzero_pos]

        plt.clf()
        plt.imshow(Image.fromarray(original_img2))  # 3d skeleton projected over the original image
        plt.show()

    return voxel_rgb


if __name__ == '__main__':
    # testing
    import pickle

    # draw skeleton color map
    hist_x = np.arange(connection_count) + 0.5  # 0.5, 1.5, ..., connection_count - 0.5
    hist_bins = np.arange(connection_count + 1)

    plt.clf()
    _, _, patches = plt.hist(hist_x, hist_bins)
    for t, color in enumerate(skeleton_colors):
        patches[t].set_facecolor(color)
        i, j = connections[t]
        patches[t].set_label('%s+%s' % (keypoint_names[i], keypoint_names[j]))

    plt.legend()
    plt.show()

    # draw depth color map
    colormap_gradient = np.linspace(0.0, 1.0, 256)
    colormap_gradient = np.vstack((colormap_gradient, colormap_gradient))

    plt.clf()
    plt.imshow(colormap_gradient, cmap=cm.get_cmap(depth_color_map))
    plt.axis('off')
    plt.show()

    image_dir = r'D:\Work\human36m\cropped_images'
    sample_count = 20
    depth_voxel = 10

    os.chdir(image_dir)
    for clip in os.listdir('.'):
        with open(os.path.join(clip, 'metadata.pkl'), 'rb') as f_in:
            metadata_list = pickle.load(f_in)

        image_count = len(metadata_list)
        metadata_list.sort(key=lambda x: x[0])

        for i in range(sample_count):
            idx = int(round(float(i) * (image_count - 1) / (sample_count - 1)))
            file_name, bbox, keypoints_img, depth = metadata_list[idx]
            keypoints_img = np.round(keypoints_img).astype(np.int64)

            fullname = os.path.join(clip, file_name)
            print(fullname)

            img = Image.open(fullname).convert('RGB')
            w, h = img.size

            # flipped
            flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
            flipped_keypoints = keypoints_img.copy()
            flipped_keypoints[:, 0] = w - 1 - flipped_keypoints[:, 0]

            # skeleton_2d
            skeleton_2d((w, h), keypoints_img, depth, False, _DEBUG=True, original_img=img)
            skeleton_2d((w, h), flipped_keypoints, depth, True, _DEBUG=True, original_img=flipped_img)

            # skeleton_rgbd
            skeleton_rgbd((w, h), keypoints_img, depth, False, _DEBUG=True, original_img=img)
            skeleton_rgbd((w, h), flipped_keypoints, depth, True, _DEBUG=True, original_img=flipped_img)

            # skeleton_3d
            skeleton_3d((w, h), depth_voxel, keypoints_img, depth, False, _DEBUG=True, original_img=img)
            skeleton_3d((w, h), depth_voxel, flipped_keypoints, depth, True, _DEBUG=True, original_img=flipped_img)
