import tensorflow as tf
import numpy as np

# a = tf.random_uniform([2,3,4],-1,1)
# b = tf.random_uniform([4,5],-1,1)
#
# c = tf.einsum('aij,jk->aik',a, b)

d = np.array([1,2,3])
e = tf.one_hot(d, 5)



tf.InteractiveSession()
print(e.eval())