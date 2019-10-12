import torch
import numpy as np
from PIL import Image, ImageDraw


def sample(population, sample_count):
    output = list()
    total_count = len(population)

    for i in range(sample_count):
        idx = int(round(float(i) * (total_count - 1) / (sample_count - 1)))
        output.append(population[idx])

    return output


def extend_box_to_square(x1, y1, x2, y2, img_w, img_h):
    w = x2 - x1
    h = y2 - y1

    if w < h:  # tall box
        delta = (h - w) / 2
        if x1 < delta:
            x2 += (2 * delta - x1)
            x1 = 0
        elif x2 + delta > img_w:
            x1 -= (2 * delta - img_w + x2)
            x2 = img_w
        else:
            x1 -= delta
            x2 += delta
        side = h

    else:  # fat box
        delta = (w - h) / 2
        if y1 < delta:
            y2 += (2 * delta - y1)
            y1 = 0
        elif y2 + delta > img_h:
            y1 -= (2 * delta - img_h + y2)
            y2 = img_h
        else:
            y1 -= delta
            y2 += delta
        side = w

    return x1, y1, x2, y2, side


def draw_bbox(img, *bboxes, line_width=5):
    img2 = img.copy()
    draw = ImageDraw.ImageDraw(img2)

    for bbox in bboxes:
        draw.rectangle(bbox, outline='black', width=line_width)
    return img2


def to_tensor(img_or_array, normalize):
    # h * w * c
    array = np.array(img_or_array).astype(np.float32)
    array = np.transpose(array, (2, 0, 1))

    if normalize:
        array /= 255.0

    t = torch.from_numpy(array)
    return t


def to_pil_image(t, normalize):
    if len(t.size()) == 4:
        t = torch.squeeze(t, dim=0)

    array = t.detach().cpu().numpy()  # c * h * w
    array = np.transpose(array, (1, 2, 0))

    if normalize:
        array *= 255.0

    array = array.astype(np.uint8)
    pil_image = Image.fromarray(array, mode='RGB')
    return pil_image
