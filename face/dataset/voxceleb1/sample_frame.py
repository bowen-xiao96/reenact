import os, sys
import shutil
from multiprocessing import Pool

source_dir = r'F:\voxceleb1\voxceleb1_frames'
dest_dir = r'F:\voxceleb1\voxceleb1_frames_sampled'

sample_count = 50
k = 6


def initialize():
    global pid
    pid = os.getpid()


def main_routine(input_var):
    i, pv_id = input_var
    video_root = os.path.join(source_dir, pv_id)

    print('%d: %d %s' % (pid, i, video_root))

    dest_root = os.path.join(dest_dir, pv_id)
    if not os.path.exists(dest_root):
        os.makedirs(dest_root)
    elif len(os.listdir(dest_root)) == sample_count:
        return

    all_frames = list()
    for clip in sorted(os.listdir(video_root), key=lambda x: int(x)):
        clip_root = os.path.join(video_root, clip)
        if os.path.isdir(clip_root):
            frames = sorted(os.listdir(clip_root), key=lambda x: int(os.path.splitext(x)[0]))
            all_frames.extend([(clip, f) for f in frames])

    frame_count = len(all_frames)
    if frame_count <= sample_count:
        sampled_frames = all_frames

    else:
        idx = [round(i * (frame_count - 1) / (sample_count - 1)) for i in range(sample_count)]
        sampled_frames = list()
        for j in idx:
            sampled_frames.append(all_frames[j])

    for clip, f in sampled_frames:
        source_filename = os.path.join(video_root, clip, f)
        dest_filename = os.path.join(dest_root, '%s_%s' % (clip, f))
        shutil.copy(source_filename, dest_filename)


if __name__ == '__main__':
    p = Pool(k, initializer=initialize)

    all_videos = list()
    for person_id in os.listdir(source_dir):
        for video_id in os.listdir(os.path.join(source_dir, person_id)):
            all_videos.append(os.path.join(person_id, video_id))

    print(len(all_videos))

    p.map(main_routine, enumerate(all_videos), chunksize=50)
