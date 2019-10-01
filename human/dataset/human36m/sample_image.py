import os, sys
import pickle
import shutil
import numpy as np

image_dir = r'D:\Work\human36m\cropped_images'
output_dir = r'D:\Work\human36m\sampled_images'

sample_count = 400

if __name__ == '__main__':
    # sample image from each clip
    os.chdir(image_dir)

    for clip in os.listdir('.'):
        with open(os.path.join(clip, 'metadata.pkl'), 'rb') as f_in:
            metadata_list = pickle.load(f_in)

        metadata_list.sort(key=lambda x: x[0])
        image_count = len(metadata_list)

        output_list = list()

        output_root = os.path.join(output_dir, clip)
        if not os.path.exists(output_root):
            os.makedirs(output_root)

        for i in range(sample_count):
            idx = int(round(float(i) * (image_count - 1) / (sample_count - 1)))
            output_list.append(metadata_list[idx])

            # copy file
            name = metadata_list[idx][0]
            shutil.copy(os.path.join(clip, name), os.path.join(output_root, name))

        with open(os.path.join(output_root, 'metadata.pkl'), 'wb') as f_out:
            pickle.dump(output_list, f_out, pickle.HIGHEST_PROTOCOL)
