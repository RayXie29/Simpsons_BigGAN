from gan import BigGAN
import tensorflow as tf
from project_config import *
import matplotlib.pyplot as plt

def main():

    gan = BigGAN(noise_dim=NOISE_DIM,
                 image_dim=IMAGE_DIM,
                 channel_width_multiplier=CHANNEL_MULTIPLIER,
                 Generator_init_size=G_INIT_SIZE
            )

    generator = gan.GeneratorNetwork()


    if GENERATOR_PRETRAIN_PATH:
        print('Load generator pretrain weights')
        generator.load_weights(GENERATOR_PRETRAIN_PATH)
    else:
        raise ValueError('No pretrain weights to load...')

    plt.figure(figsize=(12,12))
    noises = tf.random.normal([100,NOISE_DIM[0]])
    generated_images = generator(noises, training=False)
    for i, image in enumerate(generated_images):
        row = i // 10
        col = i % 10
        plt.subplot(10, 10, i+1)
        plt.imshow(image*0.5+0.5)
        plt.xticks([])
        plt.yticks([])
    plt.show()

if __name__ == '__main__':
    main()
