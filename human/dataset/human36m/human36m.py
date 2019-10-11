import os, sys
import pickle
import random

import numpy as np
from PIL import Image

import torch
from torch.utils.data import *

from . import skeleton  # import skeleton
from .dataset_utils import *  # from dataset_utils import *


class Human36m(Dataset):
    def __init__(self, root_dir, modality, return_count, image_size, normalize, shuffle, sample_count,
                 keypoint_size=5, skeleton_width=2, extend_ratio=0.2, random_flip=False, random_crop=False):
        super(Human36m, self).__init__()
        self.root_dir = root_dir
        self.modality = modality
        self.return_count = return_count
        self.image_size = image_size
        self.normalize = normalize
        self.shuffle = shuffle
        self.sample_count = sample_count

        self.keypoint_size = keypoint_size
        self.skeleton_width = skeleton_width
        self.extend_ratio = extend_ratio
        self.random_flip = random_flip
        self.random_crop = random_crop

        skeleton.keypoint_size = keypoint_size
        skeleton.skeleton_width = skeleton_width

        # load the list of all images
        self.all_videos = os.listdir(root_dir)
        self.video_count = len(self.all_videos)

        # current sample pos for each video
        self.pos = [0 for _ in range(self.video_count)]

        self.all_metadata = list()
        for video in self.all_videos:
            with open(os.path.join(root_dir, video, 'metadata.pkl'), 'rb') as f_in:
                metadata = pickle.load(f_in)

            if not shuffle or sample_count is not None:
                # fixed by the same seed
                old_state = random.getstate()
                random.seed(0)
                if sample_count is not None:
                    metadata = random.sample(metadata, sample_count)
                else:
                    random.shuffle(metadata)
                random.setstate(old_state)

            if shuffle:
                random.shuffle(metadata)

            self.all_metadata.append(metadata)

    def __len__(self):
        return self.video_count

    def __getitem__(self, item):
        video = self.all_videos[item]
        metadata = self.all_metadata[item]
        image_count = len(metadata)

        # self.pos[item], self.pos[item] + 1, ..., self.pos[item] + self.return_count - 1
        if self.pos[item] + self.return_count > image_count:
            # reset
            self.pos[item] = 0
            if self.shuffle:
                random.shuffle(metadata)

        images = list()
        skeletons = list()

        # read images
        for i in range(self.return_count):
            idx = self.pos[item] + i
            file_name, bbox, keypoints_img, depth = metadata[idx]

            img_fullname = os.path.join(self.root_dir, video, file_name)
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

                new_x1 = x1 - extend_left
                new_y1 = y1 - extend_up
                new_x2 = x2 + extend_right
                new_y2 = y2 + extend_down

            else:
                extend_ratio = self.extend_ratio / 2  # for one side
                extend = min(left, right, up, down)
                extend = min(extend, side * extend_ratio)

                new_x1 = x1 - extend
                new_y1 = y1 - extend
                new_x2 = x2 + extend
                new_y2 = y2 + extend

            # calculate new coordinates
            keypoints_img_new = keypoints_img.copy()
            keypoints_img_new[:, 0] -= new_x1
            keypoints_img_new[:, 1] -= new_y1

            # crop the image
            cropped_img = img.crop((new_x1, new_y1, new_x2, new_y2))
            new_w, new_h = cropped_img.size

            # random flip
            indicator = random.random()
            if indicator >= 0.5:
                flip = True
                cropped_img = cropped_img.transpose(Image.FLIP_LEFT_RIGHT)
                keypoints_img_new[:, 0] = new_w - 1 - keypoints_img_new[:, 0]
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
            keypoints_img_new[:, 0] *= ratio_w
            keypoints_img_new[:, 1] *= ratio_h

            if self.modality == 'skeleton_2d':
                skeleton_img = skeleton.skeleton_2d(output_size, keypoints_img_new, depth, flip)
                skeleton_img = to_tensor(skeleton_img, self.normalize)

                skeletons.append(skeleton_img)
            elif self.modality == 'skeleton_rgbd':
                skeleton_img, depth_img = skeleton.skeleton_rgbd(output_size, keypoints_img_new, depth, flip)
                skeleton_img = to_tensor(skeleton_img, self.normalize)

                depth_img = torch.from_numpy(depth_img)
                depth_img = torch.unsqueeze(depth_img, dim=0)
                skeleton_img = torch.cat((skeleton_img, depth_img), dim=0)

                skeletons.append(skeleton_img)
            else:
                raise NotImplementedError()

        # move forward pointer
        self.pos[item] += self.return_count

        # output
        skeleton_t = skeletons[0]
        image_t = images[0]
        skeletons = torch.stack(skeletons[1:])
        images = torch.stack(images[1:])

        return item, images, skeletons, image_t, skeleton_t


if __name__ == '__main__':
    # testing
    import matplotlib.pyplot as plt

    image_dir = r'D:\Work\human36m\sampled_images'
    modality = 'skeleton_rgbd'
    return_count = 3
    image_size = 256
    normalize = True
    extend_ratio = 0.2
    random_flip = True
    random_crop = True

    # including the ground truth image
    return_all = return_count + 1

    dataset = Human36m(image_dir, modality, return_all, image_size, normalize, 5, 2,
                       extend_ratio=0.2, random_flip=random_flip, random_crop=random_crop)

    print(len(dataset))

    for i, (item, images, skeletons, image_t, skeleton_t) in enumerate(dataset):
        assert i == item

        assert images.size(0) == return_count
        assert skeletons.size(0) == return_count

        image_t = torch.unsqueeze(image_t, dim=0)
        images = torch.cat((image_t, images), dim=0)
        skeleton_t = torch.unsqueeze(skeleton_t, dim=0)
        skeletons = torch.cat((skeleton_t, skeletons), dim=0)

        # image
        skeletons_rgb = skeletons[:, :-1, ...]
        canvas = Image.new('RGB', (return_all * image_size, 3 * image_size), 'white')
        for j in range(return_all):
            # original RGB image
            ii = to_pil_image(images[j], normalize)
            canvas.paste(ii, (j * image_size, 0))

            # RGB skeleton
            ss = to_pil_image(skeletons_rgb[j], normalize)
            canvas.paste(ss, (j * image_size, image_size))

            # skeleton drawn over the original image
            ii_arr = np.array(ii)
            ss_arr = np.array(ss)
            nonzero_pos = np.nonzero(np.sum(ss_arr, axis=-1))
            fused_arr = ii_arr.copy()
            fused_arr[nonzero_pos] = ss_arr[nonzero_pos]
            fused = Image.fromarray(fused_arr)
            canvas.paste(fused, (j * image_size, 2 * image_size))

        plt.clf()
        plt.imshow(canvas)
        plt.show()

        # depth
        skeletons_depth = skeletons[:, -1, ...]
        depth_canvas = np.zeros((image_size, return_all * image_size), dtype=np.float32)
        for j in range(return_all):
            depth_canvas[..., j * image_size: (j + 1) * image_size] = skeletons_depth[j].numpy()

        plt.clf()
        plt.imshow(depth_canvas, cmap=skeleton.skeleton_colormap)
        plt.colorbar()
        plt.show()
