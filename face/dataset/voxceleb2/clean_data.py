import os, sys
import shutil

root_dir = r'E:\talking_heads\data\voxceleb2\vox2_test_mp4'
recycle_dir = r'E:\talking_heads\data\voxceleb2\recycle'

if __name__ == '__main__':
    if not os.path.exists(recycle_dir):
        os.makedirs(recycle_dir)

    for person_id in os.listdir(root_dir):
        for video_id in os.listdir(os.path.join(root_dir, person_id)):
            print('%s %s' % (person_id, video_id))

            video_root = os.path.join(root_dir, person_id, video_id)
            for video in os.listdir(video_root):
                fullname = os.path.join(video_root, video)

                if os.path.getsize(fullname) < 1024:
                    shutil.move(fullname, os.path.join(recycle_dir, '%s_%s_%s' % (person_id, video_id, video)))

            if len(os.listdir(video_root)) == 0:
                shutil.move(video_root, recycle_dir)
