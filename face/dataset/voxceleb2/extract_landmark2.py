import os, sys
import pickle
import numpy as np
import multiprocessing as mp

from PIL import Image

from landmark import get_detector, extract_landmark

local_root = r'E:\talking_heads\data\voxceleb2\vox2_dev_frames_sampled'
remote_root = r'/userhome/35/ljliu/talking_heads/data/vox2_dev_frames_sampled'

process_count = 8
gpu_process = 6

frame_per_video = 20
image_size = 256


def local_routine():
    all_videos = list()

    for person_id in os.listdir(local_root):
        for video_id in os.listdir(os.path.join(local_root, person_id)):
            video_root = os.path.join(local_root, person_id, video_id)
            print(video_root)

            if os.path.isdir(video_root):
                all_videos.append((person_id, video_id))

    with open('all_videos.pkl', 'wb') as f_out:
        pickle.dump(all_videos, f_out, pickle.HIGHEST_PROTOCOL)


def remote_initialize(queue):
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


def remote_routine(input_var):
    i, (person_id, video_id) = input_var

    video_root = os.path.join(remote_root, person_id, video_id)
    if os.path.exists(os.path.join(video_root, 'metadata.pkl')):
        return True

    else:
        frames = os.listdir(video_root)
        if len(frames) != frame_per_video:
            return False
        else:
            metadata = list()

            print('%d: %d %s' % (pid, i, video_root))

            for f in frames:
                full_path = os.path.join(video_root, f)

                img = Image.open(full_path).convert('RGB')
                img = img.resize((image_size, image_size), Image.LANCZOS)

                arr = np.array(img)
                landmarks = extract_landmark(detector, arr)

                if landmarks is None:
                    print('No face detected in image' + full_path)
                    continue  # neglect this frame
                else:
                    metadata.append((f, landmarks))

            with open(os.path.join(video_root, 'metadata.pkl'), 'wb') as f_out:
                pickle.dump(metadata, f_out, pickle.HIGHEST_PROTOCOL)

            return True


if __name__ == '__main__':
    assert len(sys.argv) > 1
    mode = sys.argv[1].strip()

    if mode == 'local':
        local_routine()

    elif mode == 'remote':
        manager = mp.Manager()
        queue = manager.Queue()
        for i in range(process_count):
            queue.put(i)

        p = mp.Pool(process_count, initializer=remote_initialize, initargs=(queue, ))

        with open('all_videos.pkl', 'rb') as f_in:
            all_videos = pickle.load(f_in)

        while len(all_videos) > 0:
            finished = p.map(remote_routine, enumerate(all_videos), chunksize=50)

            all_videos = [all_videos[i] for i in range(len(all_videos)) if not finished[i]]
