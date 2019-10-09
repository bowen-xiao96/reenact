import os, sys
import pickle
import shutil

image_dir = r'D:\Work\human36m\cropped_images'
output_root = r'D:\Work\human36m'
train_sample = 400
test_sample = 18

test_actor = 11

if __name__ == '__main__':
    train_dir = os.path.join(output_root, 'train_images')
    test_dir = os.path.join(output_root, 'test_images')

    train_video_count = 0
    test_video_count = 0

    # scan all videos
    os.chdir(image_dir)

    for video in os.listdir('.'):
        print(video)

        if video.startswith('s_%d' % test_actor):
            # test
            test_video_count += 1
            output_dir = test_dir
            sample_count = test_sample
        else:
            train_video_count += 1
            output_dir = train_dir
            sample_count = train_sample

        with open(os.path.join(video, 'metadata.pkl'), 'rb') as f_in:
            metadata = pickle.load(f_in)

        metadata.sort(key=lambda x: x[0])
        image_count = len(metadata)

        output_video_root = os.path.join(output_dir, video)
        if not os.path.exists(output_video_root):
            os.makedirs(output_video_root)

        # sample
        output_metadata = list()

        for i in range(sample_count):
            idx = int(round(float(i) * (image_count - 1) / (sample_count - 1)))
            output_metadata.append(metadata[idx])

            # copy file
            name = metadata[idx][0]
            shutil.copy(os.path.join(video, name), os.path.join(output_video_root, name))

        with open(os.path.join(output_video_root, 'metadata.pkl'), 'wb') as f_out:
            pickle.dump(output_metadata, f_out, pickle.HIGHEST_PROTOCOL)

    print(train_video_count + test_video_count)
    print(train_video_count)
    print(test_video_count)
