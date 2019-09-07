# calculate metadata of the dataset and extract face landmark images
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image, ImageDraw

from skimage import io
from face_alignment import FaceAlignment, LandmarksType, NetworkSize


def get_detector(device):
    # device: 'cpu' or 'cuda'
    detector = FaceAlignment(LandmarksType._2D, network_size=NetworkSize.LARGE, device=device, flip_input=True)
    return detector


def extract_landmark(detector, image_arr):
    landmarks = detector.get_landmarks(image_arr)

    if landmarks is None:
        return None
    elif len(landmarks) == 1:
        return landmarks[0]
    else:
        # find out the main face
        # landmarks: 68 * 2 array

        ww = [np.max(arr[:, 0]) - np.min(arr[:, 0]) + 1 for arr in landmarks]
        hh = [np.max(arr[:, 1]) - np.min(arr[:, 1]) + 1 for arr in landmarks]
        area = np.array(ww) * np.array(hh)
        idx = np.argmax(area)
        return landmarks[idx]


def plot_landmarks(image_size, landmarks, original_image=None):
    # if original_image is not None, plot the landmarks over it
    if original_image is None:
        img = Image.new('RGB', (image_size, image_size), 'white')
    else:
        img = original_image.copy()

    # draw landmarks
    draw = ImageDraw.Draw(img)

    # face
    draw.line(landmarks[0:17, :], fill='green', width=3)
    # eyebrows (left and right)
    draw.line(landmarks[17:22, :], fill='orange', width=3)
    draw.line(landmarks[22:27, :], fill='orange', width=3)
    # nose (nose and nostril)
    draw.line(landmarks[27:31, :], fill='blue', width=3)
    draw.line(landmarks[31:36, :], fill='blue', width=3)
    # eyes (left and right)
    draw.line(landmarks[36:42, :], fill='red', width=3)
    draw.line(landmarks[42:48, :], fill='red', width=3)
    # lips
    draw.line(landmarks[48:60, :], fill='purple', width=3)
    # teeth
    # draw.line(landmarks[60:68, :], fill='purple', width=3)

    return img


if __name__ == '__main__':
    # for testing

    test_filename = r'D:\Personnal Stuff\Library\Desktop\HKU\data\face_align\test.jpg'
    output_filename = r'D:\Personnal Stuff\Library\Desktop\HKU\data\face_align\test_output.jpg'

    detector = get_detector('cpu')
    in_image = io.imread(test_filename)
    landmarks = detector.get_landmarks(in_image)
    landmarks = landmarks[0]

    img = Image.open(test_filename)
    output_img = plot_landmarks(img.size, landmarks, original_image=img)
    output_img.save(output_filename)
