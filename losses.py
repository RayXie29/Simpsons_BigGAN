import tensorflow as tf

def RaLS_generator_loss(real_validity, fake_validity):
    loss_1 = tf.reduce_mean( (real_validity-tf.reduce_mean(fake_validity, axis=0)+tf.ones_like(real_validity))**2, axis=0 )
    loss_2 = tf.reduce_mean( (fake_validity-tf.reduce_mean(real_validity, axis=0)-tf.ones_like(fake_validity))**2, axis=0 )
    return loss_1 + loss_2

def RaLS_discriminator_loss(real_validity, fake_validity):

    loss_1 = tf.reduce_mean( (real_validity-tf.reduce_mean(fake_validity, axis=0)-tf.ones_like(real_validity))**2, axis=0 )
    loss_2 = tf.reduce_mean( (fake_validity-tf.reduce_mean(real_validity, axis=0)+tf.ones_like(fake_validity))**2, axis=0 )
    return loss_1 + loss_2

def discriminator_hinge_loss(real_validity, fake_validity):

    real_loss = tf.reduce_mean( tf.nn.relu(1.0-real_validity), axis=0)
    fake_loss = tf.reduce_mean( tf.nn.relu(1.0+fake_validity), axis=0)
    return real_loss + fake_loss

def generator_hinge_loss(fake_validity):
    return -tf.reduce_mean(fake_validity, axis=0)
