import os
import yaml


BAD_IMAGE_FILENAMES = [
  '8524.png', '5617.png', '7958.png', '4791.png', '9541.png', '3889.png', '9684.png', '6692.png',
  '5629.png', '73.png',   '191.png',  '242.png' , '314.png',  '388.png',  '660.png',  '678.png',
  '684.png',  '697.png',  '736.png',  '737.png',  '772.png',  '773.png',  '803.png',  '1103.png',
  '1272.png', '1281.png'
]


BATCH_SIZE=64
EPOCHS=1000
NOISE_DIM=(128,)
IMAGE_DIM=(64, 64, 3)

GENERATOR_PRETRAIN_PATH = './checkpoint/generator.hdf5'
GENERATOR_CHECKPOINT_PATH = './checkpoint/generator.hdf5'
DISCRIMINATOR_PRETRAIN_PATH = './checkpoint/generator.hdf5'
DISCRIMINATOR_CHECKPOINT_PATH = './checkpoint/generator.hdf5'


DISPLAY_STEP = 75
C_IMGS_DIR = './train_images/cropped'
CHANNEL_MULTIPLIER = 16
G_INIT_SIZE = (8,8)
G_LR = 1e-4
D_LR = 2e-4
