import tensorflow as tf

def Orthogonal_Regularization_Relax(beta):

    def func(w):
        w_shape = w.get_shape().list()
        c = w_shape[-1]
        w = tf.reshape(w, [-1,c])
        mat = tf.ones_like(tf.eye(c)) - tf.eye(c)
        w_t = tf.transpose(w)

        w_mul = tf.matamul(w_t, w)
        reg = tf.subract(w_mul-mat)
        reg_loss = tf.nn.l2_loss(reg)

        return beta * reg_loss

    return func
