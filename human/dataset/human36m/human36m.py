import os, sys
import pickle
import random

import numpy as np
from PIL import Image

import torch
from torch.utils.data import *

import dataset.human36m.skeleton as skeleton
from .dataset_utils import *


class Human36m(Dataset):
    def __init__(self, root_dir, modality, return_count, image_size, normalize, keypoint_size=5, skeleton_width=2,
                 extend_ratio=0.2, random_flip=False, random_crop=False):
        super(Human36m, self).__init__()
        self.root_dir = root_dir
        self.modality = modality
        self.return_count = return_count
        self.image_size = image_size
        self.normalize = normalize
        self.keypoint_size = keypoint_size
        self.skeleton_width = skeleton_width

        self.extend_ratio = extend_ratio
        self.random_flip = random_flip
        self.random_crop = random_crop

        skeleton.keypoint_size = keypoint_size
        skeleton.skeleton_width = skeleton_width

        # load the list of all images
        clip_list = os.listdir(root_dir)
        self.clip_list = clip_list
        self.clip_count = len(clip_list)

        self.pos = [0 for _ in range(self.clip_count)]

        video_list = list()
        for clip in clip_list:
            with open(os.path.join(root_dir, clip, 'metadata.pkl'), 'rb') as f_in:
                metadata_list = pickle.load(f_in)

            random.shuffle(metadata_list)
            video_list.append(metadata_list)

        self.video_list = video_list

    def __len__(self):
        return self.clip_count

    def __getitem__(self, item):
        clip = self.clip_list[item]
        metadata_list = self.video_list[item]
        image_count = len(metadata_list)

        if self.pos[item] + self.return_count > image_count:
            # reset
            self.pos[item] = 0
            random.shuffle(metadata_list)

        images = list()
        skeletons = list()

        # read images
        for i in range(self.return_count):
            # [pos, pos + 1, ... pos + return_count - 1]
            file_name, bbox, keypoints_img, depth = metadata_list[self.pos[item] + i]

            img_fullname = os.path.join(self.root_dir, clip, file_name)
            img = Image.open(img_fullname).convert('RGB')
            w, h = img.size

            # get base box
            x1, y1, x2, y2 = bbox

            # extend the box to square
            x1, y1, x2, y2, side = extend_box_to_square(x1, y1, x2, y2, w, h)

            # space left on four sides
            left = x1
            right = w - x2
            up = y1
            down = h - y2

            if self.random_crop:
                max_extend = min(left + right, up + down)
                max_extend = min(max_extend, side * self.extend_ratio)
                extend = random.uniform(0, max_extend)

                extend_left = random.uniform(max(0, extend - right), min(left, extend))
                extend_right = extend - extend_left
                extend_up = random.uniform(max(0, extend - down), min(up, extend))
                extend_down = extend - extend_up

                x1 -= extend_left
                y1 -= extend_up
                x2 += extend_right
                y2 += extend_down
                side += extend

            else:
                extend_ratio = float(self.extend_ratio) / 2  # for one side
                extend = min(left, right, up, down)
                extend = min(extend, side * extend_ratio)

                x1 -= extend
                y1 -= extend
                x2 += extend
                y2 += extend
                side += 2 * extend

            # calculate new coordinates
            keypoints_img[:, 0] -= x1
            keypoints_img[:, 1] -= y1

            # crop the image
            cropped_img = img.crop((x1, y1, x2, y2))
            new_w, new_h = cropped_img.size

            # random flip
            indicator = random.random()
            if indicator >= 0.5:
                flip = True
                cropped_img = cropped_img.transpose(Image.FLIP_LEFT_RIGHT)
                keypoints_img[:, 0] = new_w - 1 - keypoints_img[:, 0]
            else:
                flip = False

            # resize
            ratio_w = self.image_size / new_w
            ratio_h = self.image_size / new_h
            output_size = (self.image_size, self.image_size)
            cropped_img = cropped_img.resize(output_size, Image.LANCZOS)

            # convert to tensor
            images.append(to_tensor(cropped_img, self.normalize))

            # calculate skeleton
            keypoints_img[:, 0] *= ratio_w
            keypoints_img[:, 1] *= ratio_h

            if self.modality == 'skeleton_2d':
                skeleton_img = skeleton.skeleton_2d(output_size, keypoints_img, depth, flip)
                skeleton_img = to_tensor(skeleton_img, self.normalize)

                skeletons.append(skeleton_img)
            elif self.modality == 'skeleton_rgbd':
                skeleton_img, depth_map = skeleton.skeleton_rgbd(output_size, keypoints_img, depth, flip)
                skeleton_img = to_tensor(skeleton_img, self.normalize)

                depth_map = torch.from_numpy(depth_map)
                depth_map = torch.unsqueeze(depth_map, dim=0)
                skeleton_img = torch.cat((skeleton_img, depth_map), dim=0)

                skeletons.append(skeleton_img)
            else:
                raise NotImplementedError()

        # move forward pointer
        self.pos[item] += self.return_count

        # form output
        skeleton_t = skeletons[0]
        image_t = images[0]
        skeletons = torch.stack(skeletons[1:])
        images = torch.stack(images[1:])

        return item, images, skeletons, image_t, skeleton_t


if __name__ == '__main__':
    # testing
    import matplotlib.pyplot as plt

    image_dir = r'D:\Work\human36m\sampled_images'
    modality = 'skeleton_2d'
    return_count = 3
    image_size = 256
    normalize = True
    extend_ratio = 0.2
    random_flip = True
    random_crop = True

    # including the ground truth image
    return_all = return_count + 1

    dataset = Human36m(image_dir, modality, return_all, image_size, normalize, 5, 2,
                       extend_ratio, random_flip=random_flip, random_crop=random_crop)

    print(len(dataset))

    for i in range(5):
        for j, (item, images, skeletons, image_t, skeleton_t) in enumerate(dataset):
            assert j == item
            print('%d: %d' % (i, j))

            assert images.size(0) == return_count
            assert skeletons.size(0) == return_count
            images = torch.cat((torch.unsqueeze(image_t, dim=0), images), dim=0)
            skeletons = torch.cat((torch.unsqueeze(skeleton_t, dim=0), skeletons), dim=0)

            output_img = Image.new('RGB', (return_all * image_size, 3 * image_size), 'white')
            for t in range(return_all):
                ii = to_pil_image(images[t], normalize)
                output_img.paste(ii, (t * image_size, 0))

                ss = to_pil_image(skeletons[t], normalize)
                output_img.paste(ss, (t * image_size, image_size))

                ii_arr = np.array(ii)
                ss_arr = np.array(ss)
                nonzero_pos = np.nonzero(np.sum(ss_arr, axis=-1))
                fused = ii_arr.copy()
                fused[nonzero_pos] = ss_arr[nonzero_pos]
                fused = Image.fromarray(fused)
                output_img.paste(fused, (t * image_size, 2 * image_size))

            plt.clf()
            plt.imshow(output_img)
            plt.show()
