import os, sys
import numpy as np
from multiprocessing import Pool

from PIL import Image
import skvideo

skvideo.setFFmpegPath(r'C:\ffmpeg-4.2-win64-shared\bin')
import skvideo.io

root_dir = r'E:\talking_heads\data\voxceleb2\vox2_dev_mp4'
output_dir = r'E:\talking_heads\data\voxceleb2\vox2_dev_frames'
k = 6
sample_count = 20


def initialize():
    global pid
    pid = os.getpid()


def main_routine(input_var):
    i, pv_id = input_var
    video_root = os.path.join(root_dir, pv_id)
    output_root = os.path.join(output_dir, pv_id)
    videos = os.listdir(video_root)

    print('%d: %d %s %d' % (pid, i, video_root, len(videos)))

    # extract
    for video in videos:
        video_name, _ = os.path.splitext(video)
        frame_path = os.path.join(output_root, video_name)  # output directory

        if os.path.exists(frame_path):
            if len(os.listdir(frame_path)) == sample_count:
                break
        else:
            os.makedirs(frame_path)

        fullname = os.path.join(video_root, video)

        # read video
        vid_reader = skvideo.io.FFmpegReader(fullname)
        frame_count, _, _, _ = vid_reader.getShape()

        # calculate indices
        idx = [round(i * (frame_count - 1) / (sample_count - 1)) for i in range(sample_count)]

        for i, frame in enumerate(vid_reader.nextFrame()):
            if i in idx:
                img = Image.fromarray(frame)
                img.save(os.path.join(frame_path, '%d.jpg' % i))

        vid_reader.close()


if __name__ == '__main__':
    p = Pool(k, initializer=initialize)

    all_videos = list()
    for person_id in os.listdir(root_dir):
        for video_id in os.listdir(os.path.join(root_dir, person_id)):
            all_videos.append(os.path.join(person_id, video_id))

    print(len(all_videos))

    p.map(main_routine, enumerate(all_videos), chunksize=50)
