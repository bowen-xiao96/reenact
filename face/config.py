# path
train_dataset = '/userhome/35/ljliu/talking_heads/data/voxceleb1/data'
test_dataset = '/userhome/35/ljliu/talking_heads/data/voxceleb1/data'

vgg19_weight_file = '/userhome/35/ljliu/talking_heads/weight/vgg19_caffe.pth'
vggface_weight_file = '/userhome/35/ljliu/talking_heads/weight/vggface_caffe.pth'

# gpu
gpu_id = 0

# dataset
dataset_shuffle = True
num_worker = 4
random_flip = True
input_size = 256
input_normalize = True

# checkpoint
max_epoch = 50
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

# layer config for the three networks
embedding_dim = 512  # structure of the embedder

G_config = [
    ('D', 64, 9, 2),  # 128
    ('I', ),
    ('D', 128, 3, 2),  # 64
    ('I',),
    ('D', 256, 3, 2),  # 32
    ('I',),
    ('A', ),  # operate at 32 x 32
    ('D', 512, 3, 2),  # 16
    ('I',),
    ('BA', 512, 3),
    ('BA', 512, 3),
    ('BA', 512, 3),
    ('BA', 512, 3),
    ('BA', 512, 3),
    ('U', 256, 3, 2),  # 32
    ('U', 128, 3, 2),  # 64
    ('A', 8),  # operate at 64 x 64
    ('U', 64, 3, 2),  # 128
    ('U', 3, 3, 2),  # 256
]

# for embedder, do not use normalization layers
E_config = [
    ('D', 64, 3, 2),  # 128
    ('D', 128, 3, 2),  # 64
    ('D', 256, 3, 2),  # 32
    ('A', ),  # operate at 32 x 32
    ('D', 512, 3, 2),  # 16
    ('D', 512, 3, 2),  # 8
    ('D', 512, 3, 2),  # 4 x 4 spatial resolution
]

# for discriminator, do not use normalization layers
V_config = [
    ('D', 64, 3, 2),  # 128
    ('D', 128, 3, 2),  # 64
    ('D', 256, 3, 2),  # 32
    ('A', ),  # operate at 32 x 32
    ('D', 512, 3, 2),  # 16
    ('D', 512, 3, 2),  # 8
    ('D', 512, 3, 2),  # 4
    ('B', 512, 3)  # 4 x 4 spatial resolution
]

upsample_mode = 'bilinear'

# layers of the VGG19 and VGGFace networks used to compute content loss
vgg19_layers = (0, 5, 10, 19, 28)  # before relu activation
vggface_layers = (0, 5, 10, 17, 24)

# weight of each loss
vgg19_loss_weight = 1e-2
vggface_loss_weight = 2e-3
fm_loss_weight = 1e1
mch_loss_weight = 8e1

# learning rate
lr_EG = 5e-5
lr_D = 2e-4

adam_betas = (0.0, 0.999)
