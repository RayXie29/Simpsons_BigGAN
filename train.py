import time
from losses import *
from gan import BigGAN
import tensorflow as tf
from data import dataset
from project_config import *
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt

def train_step_both(images):

    noise = tf.random.truncated_normal(shape=[BATCH_SIZE, NOISE_DIM[0]])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_hinge_loss(fake_output)
        disc_loss = discriminator_hinge_loss(real_output, fake_output)

    gen_grad = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_grad = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    G_optimizer.apply_gradients(zip(gen_grad, generator.trainable_variables))
    D_optimizer.apply_gradients(zip(disc_grad, discriminator.trainable_variables))

    return gen_loss.numpy(), disc_loss.numpy()

def display_generated_images(model, display_num):
    noises = tf.random.normal([display_num, NOISE_DIM[0]])
    generated_images = model(noises, training=False)
    ROW = display_num/5 + 1 if (display_num%5 != 0) else display_num/5
    COL = 5
    plt.figure(figsize=(int(ROW*6), 6))
    for idx, image in enumerate(generated_images):
        image = image * 0.5 + 0.5
        plt.subplot(ROW, COL , idx+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(None)
        plt.imshow(image)
    plt.show()

def train(train_dataset, steps_per_epoch):

    for e in range(EPOCHS):
        start = time.time()
        for step in tqdm_notebook(range(0, steps_per_epoch, 1)):
            input_batch = train_dataset.next()
            gen_loss, disc_loss = train_step_both(input_batch)

            if step%DISPLAY_STEP == 0:
                print('Epoch : {}, Generator loss : {}, Discriminator loss : {}'.format(e+1, gen_loss, disc_loss))

        display_generated_images(generator, 5)
        print('Spending time : {}'.format(time.time()-start))


def main():

    ds = dataset(batch_size=BATCH_SIZE, image_dim=IMAGE_DIM, file_path = C_IMGS_DIR)
    train_dataset = ds.GetDataset()

    tf.keras.backend.clear_session()

    gan = BigGAN(noise_dim=NOISE_DIM,
                 image_dim=IMAGE_DIM,
                 channel_width_multiplier=CHANNEL_MULTIPLIER,
                 Generator_init_size=G_INIT_SIZE
            )

    generator = gan.GeneratorNetwork()
    discriminator = gan.DiscriminatorNetwork()

    if GENERATOR_PRETRAIN_PATH:
        print('Load generator pretrain weights')
        generator.load_weights(GENERATOR_PRETRAIN_PATH)

    if DISCRIMINATOR_PRETRAIN_PATH:
        print('Load discriminator pretrain weights')
        discriminator.load_weights(DISCRIMINATOR_PRETRAIN_PATH)

    G_optimizer = tf.keras.optimizers.Adam(lr=G_LR, beta_1=0.0, beta_2=0.9)
    D_optimizer = tf.keras.optimizers.Adam(lr=D_LR, beta_1=0.0, beta_2=0.9)

    train(train_dataset, int(ds.__len__()))

    print('*'*20)
    print('Model training finished')
    print('Saving trained weights...')
    print('*'*20)

    generator.save_weights(GENERATOR_CHECKPOINT_PATH)
    discriminator.save_weights(DISCRIMINATOR_CHECKPOINT_PATH)

if __name__ == '__main__':
    main()
