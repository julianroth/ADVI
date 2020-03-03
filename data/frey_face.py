from scipy.io import loadmat
import tensorflow as tf

def load_data():
    mat = loadmat('data/frey_rawface')
    data = mat['ff']
    data = tf.constant(data, dtype=tf.float64)
    return tf.transpose(data)
