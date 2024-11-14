import tensorflow as tf
import numpy as np

def l2_norm(input_tensor, axis=1):
    norm = tf.norm(input_tensor, ord=2, axis=axis, keepdims=True)
    return tf.divide(input_tensor, norm)

class AdaFace(tf.keras.layers.Layer):
    def __init__(self,
                 embedding_size=512,
                 classnum=70722,
                 m=0.4,
                 h=0.333,
                 s=64.,
                 t_alpha=1.0):
        super(AdaFace, self).__init__()
        self.classnum = classnum
        self.kernel = self.add_weight("kernel", shape=(embedding_size, classnum),
                                      initializer=tf.keras.initializers.RandomUniform(minval=-1, maxval=1),
                                      trainable=True)
        self.m = m
        self.eps = 1e-3
        self.h = h
        self.s = s
        self.t_alpha = t_alpha

        # Initialize EMA
        self.t = tf.Variable(0.0, trainable=False)
        self.batch_mean = tf.Variable(20.0, trainable=False)
        self.batch_std = tf.Variable(100.0, trainable=False)

        print('\nAdaFace with the following property:')
        print('self.m:', self.m)
        print('self.h:', self.h)
        print('self.s:', self.s)
        print('self.t_alpha:', self.t_alpha)

    def call(self, embeddings, norms, labels):
        kernel_norm = l2_norm(self.kernel, axis=0)
        cosine = tf.matmul(embeddings, kernel_norm)
        cosine = tf.clip_by_value(cosine, -1 + self.eps, 1 - self.eps)

        safe_norms = tf.clip_by_value(norms, 0.001, 100)
        safe_norms = tf.stop_gradient(safe_norms)

        # Update batch mean and std
        mean = tf.reduce_mean(safe_norms)
        std = tf.math.reduce_std(safe_norms)
        self.batch_mean.assign(self.t_alpha * mean + (1 - self.t_alpha) * self.batch_mean)
        self.batch_std.assign(self.t_alpha * std + (1 - self.t_alpha) * self.batch_std)

        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std + self.eps)
        margin_scaler = margin_scaler * self.h
        margin_scaler = tf.clip_by_value(margin_scaler, -1, 1)

        # g_angular
        m_arc = tf.one_hot(labels, depth=self.classnum) * (self.m * -margin_scaler)
        theta = tf.acos(cosine)
        theta_m = tf.clip_by_value(theta + m_arc, self.eps, np.pi - self.eps)
        cosine = tf.cos(theta_m)

        # g_additive
        m_cos = tf.one_hot(labels, depth=self.classnum) * (self.m + self.m * margin_scaler)
        cosine = cosine - m_cos

        # Scale
        scaled_cosine_m = cosine * self.s
        return scaled_cosine_m
