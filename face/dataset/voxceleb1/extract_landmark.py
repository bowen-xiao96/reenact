import os, sys
import pickle
import numpy as np
import multiprocessing as mp

from PIL import Image

from landmark import get_detector, extract_landmark

root_dir = r'/userhome/35/ljliu/talking_heads/data/vox1_frames_sampled'

process_count = 6
gpu_process = 6

image_size = 256


def initialize(queue):
    global pid
    pid = os.getpid()

    global detector
    idx = queue.get()
    if idx < gpu_process:
        print('Rank: %d, PID: %d, using GPU' % (idx, pid))
        detector = get_detector('cuda')
    else:
        print('Rank: %d, PID: %d, using CPU' % (idx, pid))
        detector = get_detector('cpu')


def main_routine(input_var):
    i, pv_id = input_var

    video_root = os.path.join(root_dir, pv_id)
    if os.path.exists(os.path.join(video_root, 'metadata.pkl')):
        return True

    else:
        frames = os.listdir(video_root)
        metadata = list()

        print('%d: %d %s' % (pid, i, video_root))

        for f in frames:
            full_path = os.path.join(video_root, f)

            img = Image.open(full_path).convert('RGB')
            img = img.resize((image_size, image_size), Image.LANCZOS)

            arr = np.array(img)
            landmarks = extract_landmark(detector, arr)

            if landmarks is None:
                print('No face detected in image ' + full_path)
                continue  # neglect this frame
            else:
                metadata.append((f, landmarks))

        with open(os.path.join(video_root, 'metadata.pkl'), 'wb') as f_out:
            pickle.dump(metadata, f_out, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    all_videos = list()
    for person_id in os.listdir(root_dir):
        for video_id in os.listdir(os.path.join(root_dir, person_id)):
            all_videos.append(os.path.join(person_id, video_id))

    print(len(all_videos))

    manager = mp.Manager()
    queue = manager.Queue()
    for i in range(process_count):
        queue.put(i)

    p = mp.Pool(process_count, initializer=initialize, initargs=(queue,))

    p.map(main_routine, enumerate(all_videos), chunksize=50)
