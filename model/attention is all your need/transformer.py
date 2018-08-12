# model for machine translation and question answering

import tensorflow as tf
from Modules import *
from multi_head_attention import *



__author__ = "Xie Pan"


class Transformer():
    def __init__(self,d_k, d_v,sentence_len, d_model,
                 vocab_size_cn,
                 vocab_size_en,
                 heads=8,
                 num_layers=6,
                 is_training=True,
                 learning_rate=0.01,
                 dropout_keep_pro=0.1,
                 initializer=tf.random_normal_initializer(0,1)):

        # set hyperparameters
        self.d_k = d_k
        self.d_v = d_v
        self.sentence_len = sentence_len
        self.d_model = d_model   # the dimension of input and output
        self.vocab_size_cn = vocab_size_cn
        self.vocab_size_en = vocab_size_en
        self.heads = heads
        self.num_layers = num_layers   # the number of sub_layers
        self.is_training = is_training
        self.lr = learning_rate
        self.initializer = initializer
        self.dropout_keep_prob = dropout_keep_pro


        # add placeholders
        self.input_x = tf.placeholder(dtype=tf.int32, shape=[None, self.sentence_len])
        self.input_y = tf.placeholder(dtype=tf.int32, shape=[None, self.sentence_len])

        # define decoder inputs
        self.decoder_inputs = tf.concat([tf.ones_like(self.input_y[:,:1])*2, self.input_y[:,:-1]],axis=-1) # 2:<S>

        # encoder
        self.enc = self._encoder()

        # decoder
        self.dec = self._decoder()

        # finall linear projection
        self.logits = tf.layers.dense(self.dec, units=self.vocab_size_en) # [batch, sentence_len, vocab_size_en]
        self.prediction = tf.argmax(self.logits, axis=-1)
        self.istarget = tf.to_float(tf.not_equal(self.input_y, 0))
        self.accuracy = tf.cast(tf.equal(self.prediction, tf.argmax(self.input_y, axis=-1)), dtype=tf.float32)
        self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.prediction, self.input_y)) * self.istarget) / (
            tf.reduce_sum(self.istarget))
        tf.summary.scalar("accuracy", self.acc)


        # loss and accuracy
        if not self.is_training:
            return
        self.y_smoothed = label_smoothing(tf.one_hot(self.input_y, depth=self.vocab_size_en))
        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_smoothed)
        tf.summary.scalar("loss", self.loss)

        self.global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
        self.train_step = tf.Variable(0, trainable=False, name='train_step')
        self.train_step = tf.assign(self.train_step, tf.add(self.train_step, tf.constant(1)))
        self.train = self.add_train_op()
        tf.summary.scalar('train_step', self.train_step)


    def add_train_op(self):
        self.optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.9, beta2=0.98, epsilon=1e-9)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        return self.train_op


    def _encoder(self):
        with tf.variable_scope("encoder"):
            # 1. embedding
            with tf.variable_scope("embedding-layer"):
                self.enc = embedding(inputs=self.input_x,
                                       vocab_size=self.vocab_size_cn,
                                       num_units=self.d_model,
                                       scale=True)   # [batch, sentence_len, d_model]

            # 2. position encoding
            with tf.variable_scope("position_encoding"):
                encoding = position_encoding_mine(self.enc.get_shape()[1], self.d_model)
                self.enc *= encoding

            # 3.dropout
            self.enc = tf.layers.dropout(self.enc,
                                         rate=self.dropout_keep_prob,
                                         training=self.is_training)

            # 4. Blocks
            for i in range(self.num_layers):
                with tf.variable_scope("num_layer_{}".format(i)):
                    # multihead attention
                    # encoder: self-attention
                    self.enc = multiheadattention(q=self.enc,
                                                  k=self.enc,
                                                  v=self.enc,
                                                  d_model=self.d_model,
                                                  heads=self.heads,
                                                  causality=False,
                                                  dropout_keep_prob=self.dropout_keep_prob,
                                                  is_training=True)
                    # Feed Froward
                    self.enc = position_wise_feed_forward(self.enc,
                                                          num_units1= 4*self.d_model,
                                                          num_units2= self.d_model,
                                                          reuse=False)
        return self.enc

    def _decoder(self):
        with tf.variable_scope("decoder"):
            # embedding
            self.dec = embedding(self.decoder_inputs,
                                 vocab_size=self.vocab_size_en,
                                 num_units=self.d_model)   # [batch, sentence_len, d_model]

            # position decoding
            encoding = position_encoding_mine(self.dec.get_shape()[1], self.d_model)
            self.dec *= encoding

            # blocks
            for i in range(self.num_layers):
                with tf.variable_scope("num_layers_{}".format(i)):
                    # self-attention
                    with tf.variable_scope("self.attention"):
                        self.dec = multiheadattention(q=self.dec,
                                                      k=self.dec,
                                                      v=self.dec,
                                                      d_model=self.d_model,
                                                      heads=self.heads,
                                                      keys_mask=True,
                                                      causality=True)

                    # encoder-decoder-attention
                    with tf.variable_scope("encoder-decoder-attention"):
                        self.dec = multiheadattention(q=self.dec,
                                                      k=self.enc,
                                                      v=self.enc,
                                                      d_model=self.d_model,
                                                      heads=self.heads,
                                                      keys_mask=True,
                                                      causality=True)

                    self.dec = position_wise_feed_forward(self.dec,
                                                          num_units1= 4*self.d_model,
                                                          num_units2= self.d_model)   # [batch, sentence_len, d_model]

        return self.dec


if __name__ == "__main__":
    d_k = 64
    d_v = 64
    sentence_len = 10
    d_model = 512
    vocab_size_en = 1000
    vocab_size_cn = 1000
    transform = Transformer(d_k, d_v,sentence_len, d_model,
                 vocab_size_cn,
                 vocab_size_en,)
    mergerd = tf.summary.merge_all()
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter('./tmp/summary/train', sess.graph)
        sess.run(tf.global_variables_initializer())








