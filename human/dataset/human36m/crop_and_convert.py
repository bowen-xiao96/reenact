import os, sys
import pickle
import itertools
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

from tables import *
from dataset_utils import *

annotation_file = r'D:\Work\human36m\annotations.h5'
image_dir = r'F:\images'
output_dir = r'D:\Work\human36m\cropped_images'  # cropped images

extend_ratio = 0.2  # enable random crop

k = 8
_DEBUG = False


def main_routine(rank):
    h5file_in = open_file(annotation_file, 'r')
    table = h5file_in.root.annotations

    pid = os.getpid()
    print('%d initialized, sample count: %d' % (pid, table.nrows))

    return_list = list()

    for i, record in enumerate(table.iterrows()):
        if i % 1000 == 0:
            print('%d: %d' % (pid, i))

        if i % k != rank:
            continue

        file_name = record['file_name']
        width = record['width']
        height = record['height']

        keypoints_cam = record['keypoints_cam']
        keypoints_img = record['keypoints_img']
        bbox = record['bbox']

        # load image
        file_name = file_name.decode('utf-8')
        file_name = os.path.join(*os.path.split(file_name))  # convert separator
        input_fullname = os.path.join(image_dir, file_name)
        img = Image.open(input_fullname).convert('RGB')

        # process bbox
        x1, y1, w, h = bbox
        x2 = x1 + w
        y2 = y1 + h
        bbox = [x1, y1, x2, y2]

        if _DEBUG:
            bbox_img = draw_bbox(img, (x1, y1, x2, y2))
            plt.clf()
            plt.imshow(bbox_img)
            plt.show()

        # extend the box to square
        x1, y1, x2, y2, side = extend_box_to_square(x1, y1, x2, y2, width, height)

        if _DEBUG:
            bbox_img = draw_bbox(img, bbox, (x1, y1, x2, y2))
            plt.clf()
            plt.imshow(bbox_img)
            plt.show()

        # enlarge the box by a ratio
        extend = min(x1, width - x2, y1, height - y2)
        extend = min(extend, side * extend_ratio)

        x1 -= extend
        y1 -= extend
        x2 += extend
        y2 += extend

        # sanity check
        assert np.allclose(x2 - x1, y2 - y1)

        # transform keypoints coordinates
        keypoints_img[:, 0] -= x1
        keypoints_img[:, 1] -= y1
        depth = keypoints_cam[:, -1]

        bbox[0] -= x1
        bbox[1] -= y1
        bbox[2] -= x1
        bbox[3] -= y1

        cropped_image = img.crop((x1, y1, x2, y2))

        if _DEBUG:
            bbox_img = draw_bbox(cropped_image, bbox)
            plt.clf()
            plt.imshow(bbox_img)
            plt.show()

        # save image
        output_fullname = os.path.join(output_dir, file_name)
        cropped_image.save(output_fullname)

        bbox = np.array(bbox)

        return_list.append((file_name, bbox, keypoints_img, depth))

    h5file_in.close()
    return return_list


if __name__ == '__main__':
    if _DEBUG:
        k = 1

    # create directory structure
    h5file_in = open_file(annotation_file, 'r')
    table = h5file_in.root.annotations

    for record in table.iterrows():
        file_name = record['file_name']
        file_name = file_name.decode('utf-8')

        dir_root, _ = os.path.split(file_name)
        dir_fullname = os.path.join(output_dir, dir_root)

        if not os.path.exists(dir_fullname):
            os.makedirs(dir_fullname)

    h5file_in.close()

    # initialize multiprocessing
    p = mp.Pool(k)

    return_var = p.map(main_routine, range(k))
    return_var = list(itertools.chain.from_iterable(return_var))

    output_data = dict()

    for i, (file_name, bbox, keypoints_img, depth) in enumerate(return_var):
        dir_root, name = os.path.split(file_name)
        if dir_root not in output_data:
            output_data[dir_root] = list()

        output_data[dir_root].append((name, bbox, keypoints_img, depth))

    for k, v in output_data.items():
        with open(os.path.join(output_dir, k, 'metadata.pkl'), 'wb') as f_out:
            pickle.dump(v, f_out, pickle.HIGHEST_PROTOCOL)
