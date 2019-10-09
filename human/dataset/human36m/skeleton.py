import os, sys
import numpy as np

import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

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
    if keypoint_names[i].startswith('L_'):
        i = keypoint_names.index(keypoint_names[i].replace('L_', 'R_'))
    elif keypoint_names[i].startswith('R'):
        i = keypoint_names.index(keypoint_names[i].replace('R_', 'L_'))

    if keypoint_names[j].startswith('L_'):
        j = keypoint_names.index(keypoint_names[j].replace('L_', 'R_'))
    elif keypoint_names[j].startswith('R'):
        j = keypoint_names.index(keypoint_names[j].replace('R_', 'L_'))

    connections_flipped.append({i, j})

# color of keypoints
keypoint_2d_color = 'gray'

# colormap of the skeleton
skeleton_colormap_name = 'jet'
skeleton_colormap = cm.get_cmap(skeleton_colormap_name)
skeleton_colors = list()

for i in range(connection_count):
    key = float(i) / (connection_count - 1)  # [0, 1]
    skeleton_colors.append(skeleton_colormap(key))  # RGBA in [0, 1] range

DEPTH_INF = np.finfo(np.float32).max

# keypoint and skeleton size
keypoint_size = 10
skeleton_width = 5


def _depth_normalize(depth, basepoint='Pelvis', normalize_factor=151.32):
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

    # depth buffer
    depth_buffer = np.full((h, w), DEPTH_INF, dtype=np.float32)
    fused = np.array(img)  # h * w * c, uint8

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

        # draw the line temporarily
        img_line = Image.new('L', (w, h), 'black')  # black:0, white: 255
        draw = ImageDraw.ImageDraw(img_line)
        draw.line((x1, y1, x2, y2), fill='white', width=skeleton_width)

        # filter out the line
        nonzero_y, nonzero_x = np.nonzero(np.array(img_line))  # x: horizontal, v: vertical
        z = _interpolate_depth(nonzero_x, nonzero_y, x1, y1, z1, x2, y2, z2)

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


def _draw_legend():
    patches = list()
    for t, (i, j) in enumerate(connections):
        patch = mpatches.Patch(color=skeleton_colors[t], label='%s_%s' % (keypoint_names[i], keypoint_names[j]))
        patches.append(patch)

    plt.legend(handles=patches)


def skeleton_2d(image_size, keypoints_2d, depth, flip, _DEBUG=False, original_img=None):
    if flip:
        c = connections_flipped
    else:
        c = connections

    img = Image.new('RGB', image_size, color='black')

    normalized_depth = _depth_normalize(depth)

    img = _draw_skeleton_2d(img, c, keypoints_2d, normalized_depth)
    img = _draw_keypoints_2d(img, keypoints_2d)

    if _DEBUG:
        # draw skeleton image
        plt.clf()
        plt.imshow(img)  # skeleton
        _draw_legend()
        plt.show()

        # draw skeleton over the original image
        original_img2 = original_img.copy()
        img2 = _draw_skeleton_2d(original_img2, c, keypoints_2d, normalized_depth)
        img2 = _draw_keypoints_2d(img2, keypoints_2d)

        plt.clf()
        plt.imshow(img2)
        _draw_legend()
        plt.show()

    return img


def skeleton_rgbd(image_size, keypoints_2d, depth, flip, _DEBUG=False, original_img=None):
    if flip:
        c = connections_flipped
    else:
        c = connections

    w, h = image_size
    depth_img = np.full((h, w), DEPTH_INF, dtype=np.float32)

    normalized_depth = _depth_normalize(depth)

    # draw skeleton
    for i, j in c:
        x1, y1 = keypoints_2d[i]
        z1 = normalized_depth[i]
        x2, y2 = keypoints_2d[j]
        z2 = normalized_depth[j]

        # draw the line temporarily
        img_line = Image.new('L', image_size, 'black')  # black:0, white: 255
        draw = ImageDraw.ImageDraw(img_line)
        draw.line((x1, y1, x2, y2), fill='white', width=skeleton_width)

        # filter out the line
        nonzero_y, nonzero_x = np.nonzero(np.array(img_line))  # x: horizontal, v: vertical
        z = _interpolate_depth(nonzero_x, nonzero_y, x1, y1, z1, x2, y2, z2)

        depth_img[nonzero_y, nonzero_x] = np.minimum(depth_img[nonzero_y, nonzero_x], z)

    # draw circle keypoints
    for i in np.argsort(-normalized_depth):  # from high to low
        x, y = keypoints_2d[i]
        z = normalized_depth[i]
        radius = float(keypoint_size) / 2
        bbox = (x - radius, y - radius, x + radius, y + radius)

        img_circle = Image.new('L', image_size, 'black')
        draw = ImageDraw.ImageDraw(img_circle)
        draw.ellipse(bbox, fill='white')
        nonzero_pos = np.nonzero(np.array(img_circle))

        depth_img[nonzero_pos] = z

    depth_img[depth_img == DEPTH_INF] = 0.0

    if _DEBUG:
        # draw depth image
        plt.clf()
        plt.imshow(depth_img, cmap=skeleton_colormap)
        plt.colorbar()
        plt.show()

        # draw depth over the original image
        plt.clf()
        plt.imshow(original_img, alpha=0.6)
        plt.imshow(depth_img, cmap=skeleton_colormap, alpha=0.4)
        plt.colorbar()
        plt.show()

    skeleton_img = skeleton_2d(image_size, keypoints_2d, depth, flip, _DEBUG=_DEBUG, original_img=original_img)
    return skeleton_img, depth_img


if __name__ == '__main__':
    # testing
    import pickle

    image_dir = r'D:\Work\human36m\cropped_images'
    sample_count = 20

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

                skeleton_rgbd((w, h), keypoints_img, depth, False, _DEBUG=True, original_img=img)

                # flipped
                flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                flipped_keypoints = keypoints_img.copy()
                flipped_keypoints[:, 0] = w - 1 - flipped_keypoints[:, 0]

                skeleton_rgbd((w, h), flipped_keypoints, depth, True, _DEBUG=True, original_img=flipped_img)
