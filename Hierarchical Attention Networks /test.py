import tensorflow as tf
import numpy as np


# ？？？？？？？？
# a = tf.truncated_normal(shape=[4,3],seed=1)
# b = tf.constant(1,shape=[1,3],dtype=tf.float32)
# c = tf.multiply(a, b)
#
# tf.InteractiveSession()
# print(a.eval())
# print(b.eval())
# print(c.eval())
# # print(c.eval())
#
# # reverse
# a = [1,2,3,4,5]
# a.reverse()
#
# # tf.InteractiveSession()

a = np.random.randn(2,3,4)
b = np.array([1,2,3,4])
c = np.multiply(a, b)
print(a)
print(c)