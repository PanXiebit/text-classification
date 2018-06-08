# import tensorflow as tf
#
# a = tf.random_uniform(shape=[5,1,1,3],maxval=10)
# b = tf.squeeze(a)
# tf.InteractiveSession()
# print(b.eval())
# print(b)
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
print("tensorflow version:{0}".format(tf.__version__))
print("matplotlib version:{0}".format(matplotlib.__version__))