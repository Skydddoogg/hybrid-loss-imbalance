import tensorflow as tf
from tensorflow.python.framework import constant_op

def tf_count(t, val):
    elements_equal_to_value = tf.equal(t, val)
    as_ints = tf.cast(elements_equal_to_value, tf.int32)
    count = tf.reduce_sum(as_ints)
    return count

def _constant_to_tensor(x, dtype):
  return constant_op.constant(x, dtype=dtype)