import os, sys
import json
import numpy as np

from tables import *

annotation_dir = r'F:\annotations'
output_file = r'D:\Work\human36m\annotations.h5'


class ImageDescription(IsDescription):
    # images
    file_name = StringCol(66)
    width = UInt16Col()
    height = UInt16Col()
    subject = UInt16Col()
    cam_param_R = Float32Col((3, 3))
    cam_param_t = Float32Col(3)
    cam_param_f = Float32Col(2)
    cam_param_c = Float32Col(2)

    # annotations
    keypoints_world = Float32Col((17, 3))
    keypoints_cam = Float32Col((17, 3))
    keypoints_img = Float32Col((17, 2))
    keypoints_vis = BoolCol(17)
    bbox = Float32Col(4)


if __name__ == '__main__':
    h5file = open_file(output_file, 'w')
    table = h5file.create_table(h5file.root, 'annotations', ImageDescription)
    image_des = table.row

    # read json annotations
    for f in os.listdir(annotation_dir):
        _, ext = os.path.splitext(f)
        if ext == '.json':
            fullname = os.path.join(annotation_dir, f)
            print(fullname)

            with open(fullname, 'r') as f_in:
                j = json.load(f_in)

            for ii, aa in zip(j['images'], j['annotations']):
                image_des['file_name'] = ii['file_name']
                image_des['width'] = np.uint16(ii['width'])
                image_des['height'] = np.uint16(ii['height'])
                image_des['subject'] = np.uint16(ii['subject'])
                image_des['cam_param_R'] = np.array(ii['cam_param']['R'], dtype=np.float32)
                image_des['cam_param_t'] = np.array(ii['cam_param']['t'], dtype=np.float32)
                image_des['cam_param_f'] = np.array(ii['cam_param']['f'], dtype=np.float32)
                image_des['cam_param_c'] = np.array(ii['cam_param']['c'], dtype=np.float32)

                image_des['keypoints_world'] = np.array(aa['keypoints_world'], dtype=np.float32)
                image_des['keypoints_cam'] = np.array(aa['keypoints_cam'], dtype=np.float32)
                image_des['keypoints_img'] = np.array(aa['keypoints_img'], dtype=np.float32)
                image_des['keypoints_vis'] = np.array(aa['keypoints_vis'], dtype=np.bool_)
                image_des['bbox'] = np.array(aa['bbox'], dtype=np.float32)

                image_des.append()

        table.flush()

    h5file.close()
