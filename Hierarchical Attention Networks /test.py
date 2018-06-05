import tensorflow as tf
import numpy as np


a = tf.constant([0,1,2,3,4,5])
a = tf.split(a, 3)
b = tf.constant([5,4,3,2,1,0])
b = tf.split(b, 3)

# a = [1,2,3]
# b = [4,5,6]

c = zip(a, b) # zip object, each element is a tuple.

tf.InteractiveSession()
for i,j in zip(a,b):
    print(i.eval())
    print(j.eval())
# d = [tf.concat([i ,j], axis=1) for i,j in c]
# print(d)