import os, sys
import pickle
import random

import numpy as np
from PIL import Image

import torch
from torch.utils.data import *

from dataset.landmark import plot_landmarks
from dataset.utils import to_tensor


class VoxCeleb(Dataset):
    def __init__(self, root_dir, image_size, return_count, normalize, random_flip=False, _debug=False):
        self.root_dir = root_dir
        self.image_size = image_size
        self.return_count = return_count
        self.normalize = normalize
        self.random_flip = random_flip
        self._debug = _debug

        # load the list of all images and their frames
        print('Loading videos...')

        videos = list()
        for person_id in os.listdir(root_dir):
            for video_id in os.listdir(os.path.join(root_dir, person_id)):
                videos.append(os.path.join(person_id, video_id))

        print('Finished loading %d videos' % len(videos))

        self.videos = sorted(videos)
        self.video_count = len(videos)

    def __len__(self):
        return self.video_count

    def __getitem__(self, item):
        video = self.videos[item]
        metadata_file = os.path.join(self.root_dir, video, 'metadata.pkl')

        with open(metadata_file, 'rb') as f_in:
            frame_list = pickle.load(f_in)

        frame_count = len(frame_list)
        remains = self.return_count
        sampled_frames = list()

        while remains > frame_count:
            sampled_frames.extend(range(frame_count))
            remains -= frame_count

        sampled_frames.extend(random.sample(range(frame_count), remains))

        # sanity check
        assert len(sampled_frames) == self.return_count

        x = list()
        y = list()

        for i in sampled_frames:
            f, landmarks = frame_list[i]
            full_path = os.path.join(self.root_dir, video, f)

            img = Image.open(full_path).convert('RGB')
            img = img.resize((self.image_size, self.image_size), Image.LANCZOS)

            if self.random_flip:
                indicator = random.random()
                if indicator > 0.5:
                    # flip
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    landmarks[:, 0] = self.image_size - 1 - landmarks[:, 0]

            x.append(to_tensor(img, self.normalize))

            rendered = plot_landmarks(self.image_size, landmarks)
            y.append(to_tensor(rendered, self.normalize))

            # debug
            if self._debug:
                img.save('%d.jpg' % i)
                rendered.save('%d_lm.jpg' % i)
                debug_img = plot_landmarks(self.image_size, landmarks, original_image=img)
                debug_img.save('%d_bg.jpg' % i)

        x_t = x[0]
        y_t = y[0]

        x = torch.stack(x[1:])  # return_count * c * h * w
        y = torch.stack(y[1:])

        return item, x, y, x_t, y_t
