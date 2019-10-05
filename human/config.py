import os, sys

# path
train_dataset = os.path.expanduser('~/human36m/sampled_images')
test_dataset = os.path.expanduser('~/human36m/sampled_images')

# gpu
gpu_id = 0

# dataset
dataset_shuffle = True
num_worker = 8
random_flip = True
random_crop = True

# checkpoint
max_epoch = 300
display_every = 10
check_every = 1000
save_every = 5000
finetune_epoch = 150
save_optim_state = False

# training hyperparameter
metatrain_batch_size = 3
finetune_batch_size = 2
metatrain_T = 8  # the number of training images in an episode
finetune_T = 8
d_step = 2

# learning rate
lr_EG = 5e-5
lr_D = 2e-4

adam_betas = (0.0, 0.999)
