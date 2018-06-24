# import tensorflow as tf
# # from tensorflow.contrib.linalg import LinearOperatorLowerTriangular
# #
# # output = tf.random_normal([8, 10, 10])  # [batch, length_q, length_kv]
# # diag_vals = tf.ones_like(output[0, :, :]) # []
# # tril = LinearOperatorLowerTriangular(diag_vals).to_dense()
# # masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(output)[0], 1, 1])  # (h*N, T_q, T_k)
# #
# tf.InteractiveSession()
# # # print(diag_vals.eval())
# # # print(tril.eval())
# # print(masks.eval())
#
# input = tf.ones([8,10,3])
# output = tf.layers.conv1d(input, filters=5, kernel_size=1, strides=1)
# input2 = tf.ones([8,10,1,3])
# output2 = tf.layers.conv2d(input2, filters=5, kernel_size=1, strides=(1,1))
#
# print(output.get_shape()[-1])
# print(output2)
import numpy as np
import matplotlib.pyplot as plt

num_units = 100
sentence_len = 10

i = np.tile(np.expand_dims(range(num_units), 0), [sentence_len, 1]) # (100,)-> (1, 100) ->(10, 100)

pos = np.tile(np.expand_dims(range(sentence_len), 1), [1, num_units]) #(10,)-> (10, 1) -> (10, 100)

pos = np.multiply(pos, 1/10000.0)
i = np.multiply(i, 2.0/num_units)

matrix = np.power(pos, i)

matrix[:, 1::2] = np.sin(matrix[:, 1::2])
matrix[:, ::2] = np.cos(matrix[:, ::2])

im = plt.imshow(matrix, aspect='auto')
plt.show()

