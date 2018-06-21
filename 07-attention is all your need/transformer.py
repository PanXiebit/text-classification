# transformer for machine translation and question answering

import tensorflow as tf
from data_utils import load_cn_vocab, load_en_vocab
from Modules import position_encoding_mine


__author__ = "Xie Pan"


class Transformer():
    def __init__(self,d_k, d_v, d_model, sentence_len, vocab_size, num_layers=6,
                 initializer = tf.random_normal_initializer(0,1)):

        # set hyperparameters
        self.d_k = d_k           # the 2th dimension of keys and queries
        self.d_v = d_v           # the 2th dimension of values
        self.d_model = d_model   # the dimension of input and output
        self.sentence_len = sentence_len
        self.vocab_size = vocab_size
        self.num_layers = num_layers   # the number of sub_layers




        self.initializer = initializer

        # 1. add placeholders
        self.add_placeholder()

        # 2. embedding-layer
        self.embedding()

        # 3. position encoding
        encoding = position_encoding_mine(self.sentence_len, self.d_model)
        self.keys_posed = encoding * self.keys_embeded    # [None, sentence_len ,d_model]
        self.queries_posed = encoding * self.queries_embeded # [None, sentence_len, d_model]
        self.values_posed = encoding * self.values_embeded   # [None, sentence_len, d_model]

        # 4. encoder

        # add placeholder

    def add_placeholder(self):
        self.keys = tf.placeholder(tf.int32, [None, self.sentence_len])
        self.queries = tf.placeholder(tf.int32, [None, self.sentence_len])
        self.values = tf.placeholder(tf.int32, [None, self.sentence_len]) # ?

    def embedding(self):
        with tf.variable_scope("embedding-layer"):
            embedding = tf.get_variable('embdding-cn', [self.vocab_size, self.d_model], initializer=self.initializer)
            self.keys_embeded = tf.nn.embedding_lookup(embedding, self.keys)   # [None, sentence_len, d_model]
            self.queries_embeded = tf.nn.embedding_lookup(embedding, self.queries) # [None, sentence_len, ]
            self.values_embeded = tf.nn.embedding_lookup(embedding, self.values) # [None, ]

    def encoder(self):
        pass





