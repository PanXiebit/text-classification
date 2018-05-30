# http://www.panxiaoxie.cn/2018/05/14/cs224d-lecture13
# -%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/#more

# TextCNN: 1. embedding layers, 2.convolutional layer 3.max-pooling 4. softmax layer
import tensorflow as tf
import numpy as np

class Config:
    def __init__(self, filter_sizes,num_filter,num_classes,label_size,vocab_size,embed_size,sequence_len,
                 learning_rate,batch_size,decay_steps,decay_rate,num_iter,is_training=True,multi_label_flag=False,
                 clip_gradients=5.0,decay_rate_big=0.50):
        self.filter_sizes = filter_sizes
        self.num_filter = num_filter
        self.num_classes = num_classes
        self.label_size = label_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.sequence_len = sequence_len
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.is_training = is_training
        self.clip_gradient = clip_gradients
        self.multi_label_flag = multi_label_flag
        self.num_iter = num_iter

class TextCNN:
    def __init__(self, config):
        # 1.set hyperparameters
        self.config = config

        # 2.add placeholder
        # 这里输入是定长的，所以如果需要的话应该 padding
        self.input_x = tf.placeholder(dtype=tf.int32, shape=[None, self.config.sentence_len], name='input_x')
        self.input_y = tf.placeholder(dtype=tf.int32, shape=[None], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # 3.inference
        self.logits = self.inference()

    def inference(self):
        # 1.get embedding of words in the sentence
        with tf.variable_scope('embedding-layer'):
            self.embedding = tf.get_variable(name='embedding',shape=[self.config.vocab_size,self.config.embed_size],
                                             initializer=tf.truncated_normal_initializer(0,1))
            embeded_words = tf.nn.embedding_lookup(self.embedding, self.input_x) # [None, sentence_len, embed_size]
            # three channels similar to the image. using the tf.nn.conv2d
            self.sentence_embedding_expanded = tf.expand_dims(embeded_words, axis=-1) # [None, sentence_len, embed_size, 1]

        # 2.loop each filter size. for each filter,
        # do:convolution-pooling layer(a.create filters,b.conv,c.batch normalization d.apply nolinearity,e.max-pooling)
        pooled_outputs = []
        for i, filter_size in enumerate(self.config.filter_sizes):
            # 这里命名的时候，参考的代码是 ''conv-pool-layer-{0}'.format(filter_size). 但这样如果有相同size的filter岂不是就重复了。
            with tf.variable_scope('conv-pool-layer-{0}'.format(i)):
                # a. create filter
                filter = tf.get_variable(name="filter-{0}".format(self.config.filter_size),
                                     shape=[filter_size, self.config.embed_size, 1, self.config.num_filters],
                                        initializer=tf.truncated_normal_initializer(0))

                # b. convolution operation
                conv = tf.nn.conv2d(self.sentence_embedding_expanded,filter,stride=[1,1,1,1],padding='VALID',name='conv')

                # c. batch normalization
                # conv, self.update_ema = self.batchnorm(conv, self.tst, self.iter, self.b1)

                # d. apply nolinearity
                b = tf.get_variable("b-%s"%filter_size, [self.config.num_filters])
                h = tf.nn.relu(tf.nn.bias_add(conv, b),"relu")

                # e. max-pooling
                pooling = tf.nn.max_pool(h, ksize=[1,self.config.sentence_len-filter_size+1,1,1],
                                         strides=[1,1,1,1],padding='VALID',name='pooling')
                pooled_outputs.append(pooling)


    def batchnorm(self,Ylogits, is_test, iteration, offset, convolutional=False):
        pass

        # return Ybn, update_moving_averages
