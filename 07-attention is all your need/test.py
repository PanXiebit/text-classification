import tensorflow as tf
from tensorflow.contrib.linalg import LinearOperatorLowerTriangular

output = tf.random_normal([8, 10, 10])  # [batch, length_q, length_kv]
diag_vals = tf.ones_like(output[0, :, :]) # []
tril = LinearOperatorLowerTriangular(diag_vals).to_dense()
masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(output)[0], 1, 1])  # (h*N, T_q, T_k)

tf.InteractiveSession()
# print(diag_vals.eval())
print(tril.eval())
print(masks.eval())