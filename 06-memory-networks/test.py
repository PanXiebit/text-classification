# import tensorflow as tf
#
# with tf.variable_scope('test'):
#     a = tf.get_variable('a', shape=[2, 3], initializer=tf.constant_initializer(2))
# with tf.variable_scope('test', reuse=True):
#     b = tf.get_variable('a', shape=[2,3], initializer=tf.constant_initializer(0))
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     assert (a == b)
#     print(tf.Graph)


# import re
#
# def tokenize(sentence):
#     """Return the tokens of a sentence including puunctuation.
#
#     :param sentence:
#     :return:
#     """
#     return [x.strip() for x in re.split('(\w+)?', sentence) if x.strip()]
#
#
# file = './data/tasks_1-20_v1-2/en/qa1_single-supporting-fact_train.txt'
#
# with open(file) as f:
#     for i in range(15):
#         line = f.readline().split(' ',1)
#         # nid, line = line.split(' ',1)
#         # print(type(nid))
#         print(line)

from itertools import chain
from six.moves import reduce
data = ([['mary', 'moved', 'to', 'the', 'bathroom'], ['john', 'went', 'to', 'the', 'hallway']], ['where', 'is', 'mary'], ['bathroom'])
s,q,a = data
print(chain.from_iterable(s))
word_set = set(list(chain.from_iterable(s)) + q + a)
# vocab = reduce(lambda x, y: x|y, word_set)
# print(vocab)
word_indx = dict((c, i+1) for i,c in enumerate(word_set))
print(word_indx)