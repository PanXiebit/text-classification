"""Example running MemN2N on a single bAbI task.
Download tasks from facebook.ai/babi """
# from __future__ import absolute_import
# from __future__ import print_function

from data_utils.tradition_data import load_task, vectorize_data
from sklearn import cross_validation, metrics
from memn2n_model import E2EMemm
from itertools import chain
from six.moves import range, reduce
from collections import defaultdict

import tensorflow as tf
import numpy as np

tf.flags.DEFINE_float("learning_rate", 0.01, "Learning rate for SGD.")
tf.flags.DEFINE_float("anneal_rate", 25, "Number of epochs between halving the learning rate.")
tf.flags.DEFINE_float("anneal_stop_epoch", 100, "Epoch number to end annealed lr schedule.")
tf.flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("evaluation_interval", 10, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 8, "Batch size for training.")
tf.flags.DEFINE_integer("hops", 3, "Number of hops in the Memory Network.")
tf.flags.DEFINE_integer("epochs", 100, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 20, "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("memory_size", 50, "Maximum size of memory.")
tf.flags.DEFINE_integer("task_id", 1, "bAbI task id, 1 <= id <= 20")
tf.flags.DEFINE_integer("random_state", None, "Random state.")
tf.flags.DEFINE_string("data_dir", "glove_wv/tasks_1-20_v1-2/en/", "Directory containing bAbI tasks")
FLAGS = tf.flags.FLAGS



class Single_task():
    def __init__(self, data):
        self.data = data

        # the max number of sentence in a single story
        self.max_story_size = max(map(len, (s for s, _, _ in self.data)))
        self.mean_story_size = int(np.mean([len(s) for s, _, _ in self.data]))

        self.sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in self.data)))
        self.query_size = max(map(len, (q for _, q, _ in self.data)))

        self.memory_size = min(FLAGS.memory_size, self.max_story_size)  # 10
        self.sentence_size = max(self.query_size, self.sentence_size)
        self.sentence_size += 1  # +1 for time words

    # get the word_to_index
    def vocab(self):
        # vocab = []
        # for s,q,a in glove_wv:
        #     vocab_ = (list(chain.from_iterable(s)) + q + a)
        #     vocab += vocab_
        # vocab = set(vocab)  ## 醉了醉了。。只有19个单词

        sequence = (set(list(chain.from_iterable(s)) + q + a) for s, q, a in self.data) # iterator, every element is set
        # set | set 是两个set叠加且去重
        vocab = sorted(reduce(lambda x, y: x | y, sequence))

        word_to_index = dict((c, i) for i, c in enumerate(vocab))
        # add time words/indexes
        for i in range(self.memory_size):
            word_to_index['time{}'.format(i + 1)] = 'time{}'.format(i + 1)  ##???

        self.vocab_size = len(word_to_index) + 1  # +1 for nil word
        return word_to_index

    def get_data(self, dataset):

        # train/validation/test sets
        word_to_index = self.vocab()

        S, Q, A = vectorize_data(dataset, word_to_index, self.sentence_size, self.memory_size)
        return S,Q,A

def train():
    train, test = load_task(FLAGS.data_dir, FLAGS.task_id)
    data = train + test
    single_task = Single_task(data)
    S, Q, A = single_task.get_data(train)
    trainS, valS, trainQ, valQ, trainA, valA = cross_validation.train_test_split(S, Q, A, test_size=.1,
                                                                                 random_state=FLAGS.random_state)

    testS, testQ, testA = single_task.get_data(test)

    n_train = trainS.shape[0]
    n_test = testS.shape[0]
    n_val = valS.shape[0]

    # hyperparameters
    sentence_len = single_task.sentence_size
    memory_size = single_task.memory_size
    vocab_size = single_task.vocab_size

    embed_size = FLAGS.embedding_size
    batch_size = FLAGS.batch_size
    hops = FLAGS.hops
    max_grad_norm = FLAGS.max_grad_norm

    model = E2EMemm(batch_size, sentence_len, memory_size, vocab_size, embed_size, hops, max_grad_norm)

    batches = zip(range(0, n_train - batch_size + 1, batch_size), range(batch_size, n_train + 1, batch_size))
    batches = [(start, end) for start, end in batches] # [(0,8), (8, 16)...(992, 1000)]

    merged = tf.summary.merge_all()
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter('./tmp/summary/train', sess.graph)
        sess.run(tf.global_variables_initializer())
        for t in range(1, FLAGS.epochs+1):
            # stepped learning rate
            if t - 1 <= FLAGS.anneal_stop_epoch:
                anneal = 2.0 ** ((t-1) // FLAGS.anneal_rate)
            else:
                anneal = 2.0 ** (FLAGS.anneal_stop_epoch // FLAGS.anneal.rate)
            lr = FLAGS.learning_rate / anneal

            np.random.shuffle(batches)
            total_cost = 0.0
            for start, end in batches:   # 一个epoch就是一个完整的训练集
                s = trainS[start:end]
                q = trainQ[start:end]
                a = trainA[start:end]

                feed_dict = {model._stories:s, model._queries:q, model._answer:a, model._lr:lr}
                _ = sess.run([model.train_op],feed_dict=feed_dict)

            feed_dict_train = {model._stories:trainS, model._queries:trainQ, model._answer:trainA, model._lr:lr}
            summary, loss, train_acc = sess.run([merged, model._loss, model.accuracy], feed_dict=feed_dict_train)

            feed_dict_val = {model._stories:valS, model._queries:valQ, model._answer:valA, model._lr:lr}
            val_acc = sess.run(model.accuracy,feed_dict=feed_dict_val)
            if t % FLAGS.evaluation_interval == 0:
                print("{0} epoch, loss:{1}, accuracy:{2}, val_acc:{3}".format(t, loss, train_acc, val_acc))
            train_writer.add_summary(summary, t)  # Adds a `Summary` protocol buffer to the event file.
        train_writer.close()


if __name__ == "__main__":
    tf.app.run()
