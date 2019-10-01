# Face Reenactment

### Description

This code follows the network architecture from [Few-Shot Adversarial Learning of Realistic Neural Talking Head Models](<https://arxiv.org/pdf/1905.08233.pdf>). The details of the building blocks are defined in **model/network.py**. Layer configurations of Generator, Embedder and Discriminator are defined in **config.py**, while `D` and `U` refers to downsampling and upsampling blocks defined in [Large Scale GAN Training for High Fidelity Natural Image Synthesis](<https://arxiv.org/pdf/1809.11096.pdf>) respectively, with all the Batch Normalizations in upsampling blocks replaced by Adaptive Instance Normalizations. ``I`` refers to regular (non-adaptive) Instance Normalization layers. `B` refers to convolutional blocks defined in [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](<https://arxiv.org/pdf/1603.08155.pdf>). `BA` refers to the same blocks with all the Batch Normalizations replaced by Adaptive Instance Normalizations. For the definition of all the loss functions, please view **model/loss.py**.

### Usage

#### Dataset Preprocessing

For VoxCeleb1 dataset, please first download pre-extracted frames from <http://www.robots.ox.ac.uk/~vgg/research/CMBiometrics/data/dense-face-frames.tar.gz>.

For VoxCeleb2 dataset, since there is no pre-extracted frames available from the dataset authors, you should download all the mp4 videos from <http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html>. The username and password are ``voxceleb1904 ``  and `9hmp7488` respectively. Some videos in the archive are corrupted, You should run **dataset/voxceleb2/clean_data.py** to remove them. After that, you should use **dataset/voxceleb2/extract_frames.py** to extract a fixed number of frames from each video clip. Please notice some of these scripts use multiprocessing to boost performance, please adjust the number of processes according to the number of your CPU cores.

To reduce computational burden, you may use **dataset/voxceleb1/sample_frame.py** to sample a fixed number of frames from each video.

After all the frames are ready, you should use **dataset/voxceleb1/extract_landmark.py** to extract facial landmarks from each frame. Make sure you have installed the [face-alignment](<https://github.com/1adrianb/face-alignment>) library using the command ``pip install face-alignment``. This script uses GPU and runs in multiprocessing, please adjust the number of processes according to GPU usage. Notice that this step is very time-consuming.

#### Model Preparation

This network uses VGG19 and VGGFace networks to calculate the perceptual loss. These networks are originally in Caffe format. You should download VGG19 model from <http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel> and <https://gist.githubusercontent.com/ksimonyan/3785162f95cd2d5fee77/raw/f02f8769e64494bcd3d7e97d5d747ac275825721/VGG_ILSVRC_19_layers_deploy.prototxt> and download VGGFace model from <https://www.robots.ox.ac.uk/~vgg/software/vgg_face/src/vgg_face_caffe.tar.gz>, and then run **vgg_model/convert_model.py** to convert them to PyTorch format.

#### Running

After modifying all the settings and paths in **config.py**, you can run **meta_train.py** to start meta-training. It will automatically save model, checkpoint and log all the metrics to file.

To run the finetuning process, you may run ``python finetune.py <pretrained_model_file>`` to run finetuning on the pretrained model. It will automatically save the output images.