
# TextCNN: 1. embedding layers, 2.convolutional layer 3.max-pooling 4. softmax layer
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers.python.layers import optimize_loss


class TextCNN:
    def __init__(self, filter_sizes,num_filter,num_classes,label_size,vocab_size,embed_size,sequence_len,
                 learning_rate,batch_size,decay_steps,decay_rate,num_iter,is_training=True,multi_label_flag=False,
                 clip_gradients=5.0):
        # 1.set hyperparameters
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
        self.clip_gradients = clip_gradients
        self.multi_label_flag = multi_label_flag
        self.num_iter = num_iter
        self.num_filter_total = self.num_filter * len(self.filter_sizes)
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

        # 2.add placeholder
        # 这里输入是定长的，所以如果需要的话应该 padding
        self.input_x = tf.placeholder(dtype=tf.int32, shape=[None, self.sequence_len], name='input_x')
        self.input_y = tf.placeholder(dtype=tf.int32, shape=[None], name='input_y')
        self.input_y_multilabels = tf.placeholder(dtype=tf.float32, shape=[None, num_classes], name="input_y_multilabels")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # 3.inference
        self.logits = self.inference()

        # 4.computer loss
        if not self.is_training:
            return
        else:
            self.loss = self.add_loss()

        # 5.using sgd to train
        self.global_step = tf.Variable(initial_value=0,trainable=False,name="global_step")
        self.epoch_step = tf.Variable(initial_value=0, trainable=False, name="epoch_step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.train_op = self.add_train_op()

        # computer accuracy



    def inference(self):
        # 1.get embedding of words in the sentence
        with tf.variable_scope('embedding-layer'):
            self.embedding = tf.get_variable(name='embedding',shape=[self.vocab_size,self.embed_size],
                                             initializer=tf.truncated_normal_initializer(0,1))
            embeded_words = tf.nn.embedding_lookup(self.embedding, self.input_x) # [None, sentence_len, embed_size]
            # three channels similar to the image. using the tf.nn.conv2d
            self.sentence_embedding_expanded = tf.expand_dims(embeded_words, axis=-1) # [None, sentence_len, embed_size, 1]

        # 2.loop each filter size. for each filter,
        # do:convolution-pooling layer(a.create filters,b.conv,c.batch normalization d.apply nolinearity,e.max-pooling)
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            # 这里命名的时候，参考的代码是 ''conv-pool-layer-{0}'.format(filter_size). 但这样如果有相同size的filter岂不是就重复了。
            with tf.variable_scope('conv-pool-layer-{0}'.format(i)):
                # a. create filter `[filter_height * filter_width * in_channels, output_channels]`= [filter_size, embed_size, 1, num_filter]
                filter = tf.get_variable(name="filter-{0}".format(filter_size),
                                     shape=[filter_size, self.embed_size, 1, self.num_filter],
                                        initializer=tf.truncated_normal_initializer(0))

                # b. convolution operation
                conv = tf.nn.conv2d(self.sentence_embedding_expanded,filter,strides=[1,1,1,1],padding='VALID',name='conv') #[batch_size,sentence_len-filter_size+1,1,num_filters]

                # c. batch normalization
                # conv, self.update_ema = self.batchnorm(conv, self.tst, self.iter, self.b1)

                # d. apply nolinearity
                b = tf.get_variable("b-{0}".format(filter_size), [self.num_filter]) # [num_filters]
                relu = tf.nn.relu(tf.nn.bias_add(conv, b),"relu") # [batch_size, sentence_len-filter_size+1,1,num_filters]

                # e. max-pooling
                pooling = tf.nn.max_pool(relu, ksize=[1,self.sequence_len-filter_size+1,1,1],
                                         strides=[1,1,1,1],padding='VALID',name='pooling')    # shape=[batch_size,1,1,num_filters]
                pooled_outputs.append(pooling)
        # 3.combine all pooled feature maps, anf flatten the feature
        self.h_pool = tf.concat(pooled_outputs,axis=3) # [batch_size, 1, 1, num_filter_total]
        self.h_pool_flat = tf.reshape(self.h_pool, [-1,self.num_filter_total]) #[batch_size, num_filter_total]
        # self.h_pool_flat = tf.squeeze(self.h_pool)

        # 4. add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, keep_prob= self.dropout_keep_prob)

        # 5. projection
        with tf.variable_scope("output"):
            w = tf.get_variable(name='weights',shape=[self.num_filter_total, self.label_size],initializer=tf.truncated_normal_initializer(0))
            b = tf.get_variable(name='bias',shape=[self.label_size],initializer=tf.constant_initializer(0))
        output = tf.matmul(self.h_drop, w) + b
        return output

    def batchnorm(self,Ylogits, is_test, iteration, offset, convolutional=False):
        pass

        # return Ybn, update_moving_averages

    def add_loss(self, l2_lambda=0.01):
        # tf.name_scope 和 tf.variable_scope()的区别
        with tf.name_scope("loss"):
            if not self.multi_label_flag:
                # 使用tf.nn.sparse_softmax_cross_entropy_with_logits,真实标签不需要转换为one-hot向量，而且其只适用于单目标分类
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logits)
            else:
                # 多分类
                losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_multilabels, logits=self.logits)
            loss = tf.reduce_mean(losses)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if "bias" not in v.name]) * l2_lambda
            loss += l2_losses

        return loss

    def add_train_op(self):
        """based on the loss, use SGD to update parameter"""
        # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
        learning_rate = tf.train.exponential_decay(self.learning_rate,self.global_step,self.decay_steps,self.decay_rate)
        train_op = optimize_loss(self.loss,self.global_step,learning_rate,optimizer='Adam',clip_gradients=self.clip_gradients)
        return train_op


def test():
    multi_label_flag = True
    textcnn = TextCNN(filter_sizes=[2,3,4],num_filter=2, num_classes=5, vocab_size=1000,
                    embed_size=100,sequence_len=5,learning_rate=0.001,batch_size=8,
                    decay_steps=100,decay_rate=0.95,num_iter=128,is_training=True,
                    multi_label_flag=multi_label_flag,label_size=5)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(textcnn.num_iter):
            input_x = np.random.randn(textcnn.batch_size, textcnn.sequence_len)
            input_x[input_x>=0] = 1
            input_x[input_x<=0] = 0
            # 分别测试单标签分类和多标签分类
            if not multi_label_flag:
                input_y = np.zeros(shape=[textcnn.batch_size])
                loss, _ = sess.run(fetches=[textcnn.loss, textcnn.train_op],
                                   feed_dict={textcnn.input_x: input_x, textcnn.input_y: input_y,textcnn.dropout_keep_prob: 0.5})
            else:
                input_y_multibels = np.zeros([textcnn.batch_size, textcnn.num_classes])
                loss, _ = sess.run(fetches=[textcnn.loss, textcnn.train_op],
                                   feed_dict={textcnn.input_x: input_x, textcnn.input_y_multilabels: input_y_multibels,
                                              textcnn.dropout_keep_prob: 0.5})


            print("{0} epoch, loss:{1}".format(i, loss))


if __name__=="__main__":
    test()
