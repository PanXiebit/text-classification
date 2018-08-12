import os
import tensorflow as tf
import numpy as np
from glove import loadWordVec


class TextCNN(object):
    def __init__(self, tokens, sentence_len, embed_size,num_classes, l2_lambda, learning_rate,
                 kernels_size, filter_nums,static, keep_prob):
        # parameters
        self.tokens = tokens
        self.sentence_len = sentence_len
        self.embed_size = embed_size
        self.num_classes = num_classes
        self.l2_lambda = l2_lambda
        self.learning_rate = learning_rate
        self.static = static
        self.keep_prob = keep_prob
        self.kernels_size = kernels_size
        self.filter_nums = filter_nums

        # add placeholder
        self.X = tf.placeholder(shape=[None, sentence_len], dtype=tf.int32, name="input_sentence")
        self.y = tf.placeholder(shape=[None], dtype=tf.int32, name='input_label')
        self.is_training = tf.placeholder(dtype=tf.bool, name="is_training")

        # embedding
        self.sent_embeded = self._embedding()
        # forward
        self.logits = self._inference()
        self.labels = tf.one_hot(self.y, depth=self.num_classes, dtype=tf.float32)

        # cross entropy
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=self.logits)
        self.loss_reg = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
                                  if "bias" not in v.name if "embedding" not in v.name]) * self.l2_lambda
        self.loss = tf.reduce_mean(self.cross_entropy) + self.loss_reg

        tf.summary.scalar("cross-entropy", tf.reduce_mean(self.cross_entropy))
        tf.summary.scalar("loss-reg-{}".format(self.l2_lambda), self.loss_reg)
        tf.summary.scalar("total-loss",self.loss)

        # accuracy
        predict = tf.cast(tf.argmax(self.logits, axis=-1,name="predict"), tf.int32)
        correct_pred = tf.cast(tf.equal(predict, self.y), tf.float32)
        self.accuracy = tf.reduce_mean(correct_pred, name="accuracy")
        tf.summary.scalar("accuracy", self.accuracy)

        # train
        self.train_op = self._train_op()


    def _embedding(self):
        # embedding layer
        with tf.variable_scope("embedding-layer", reuse=tf.AUTO_REUSE):
            if not os.path.exists("wordVec.npy"):
                wordvec = loadWordVec(self.tokens, dimension=self.embed_size)
            else:
                wordvec = np.load("wordVec.npy")
            if self.static:
                self.embedding = tf.Variable(wordvec, dtype=tf.float32, trainable=False, name="static-embedding")
            else:
                self.embedding = tf.Variable(wordvec, dtype=tf.float32, trainable=True, name="non-static-embedging")
            tf.summary.histogram("embedding", self.embedding)

            sent_embeded = tf.nn.embedding_lookup(self.embedding, self.X) # [None, length_len, embed_size]
            self.sent_embeded = tf.expand_dims(sent_embeded, axis=-1)     # [None, length_len, embed_size, 1]
        return self.sent_embeded

    def _inference(self):
        output_pooling = []
        for i,kernel_size in enumerate(self.kernels_size):
            with tf.variable_scope("layer{}".format(i)):
                conv = tf.layers.conv2d(self.sent_embeded,
                                        filters=self.filter_nums,
                                        kernel_size=[kernel_size, self.embed_size],
                                        strides=[1,1],
                                        padding='valid')
                conv = tf.layers.batch_normalization(conv)
                relu = tf.nn.relu(conv)        # [None, (sentence_len-kernel_size+1)/2, 1, filter_nums]
                dropout = tf.nn.dropout(relu, keep_prob=self.keep_prob)

                # 1-max pooling
                output_pooled = tf.layers.max_pooling2d(dropout,
                                                        pool_size=[dropout.get_shape()[1], 1],
                                                        strides=[1,1],
                                                        padding="valid")
                output_pooled_squeeae = tf.squeeze(output_pooled, axis=[1,2])  # [None, filter_nums]
            output_pooling.append(output_pooled_squeeae)           # list

        output = tf.concat(output_pooling, axis=-1, name="pool-concat")        # [None, filter_nums*len(kernel_size)]

        # softmax
        with tf.variable_scope("affine-softmax"):
            logits = tf.layers.dense(output, units=self.num_classes, name="affine") # [None, num_classes]

        return logits

    # train_op
    def _train_op(self):
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train_op = optimizer.minimize(self.loss, global_step=self.global_step)
        tf.summary.scalar("learning-rate", self.learning_rate)
        tf.summary.scalar("global-step", self.global_step)
        return train_op





