print('started...')
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import optimize_loss
import numpy as np

class Config():
    def __init__(self):
        # set hyper-paramter
        self.label_size = 10  # 文档分类的总类别数
        self.batch_size = 1
        self.num_sampled = 5
        self.learning_rate = 0.01
        self.vocab_size = 1000
        self.embed_size = 100
        self.sentence_len = 5
        self.is_training = True
        self.decay_steps = 1000
        self.decay_rate = 0.9
        self.max_label_per_example = 5
        self.l2_lambda = 0.0001
        self.num_epochs = 20
        self.print_very = 3

class fastTextMulti():
    def __init__(self):
        self.config = Config()

        # add placeholder
        self.sentence = tf.placeholder(dtype=tf.int32, shape=[None, self.config.sentence_len], name="sentence")
        # 与单标签样本的区别，这里一个样本可能对应多个标签, 所以在计算损失值时，真实值不再是 one-hot 向量
        self.lables = tf.placeholder(dtype=tf.int32, shape=[None, self.config.max_label_per_example], name='labels')
        # 转化为multi-hot向量
        self.labels_l1999 = tf.placeholder(tf.int64, [None, self.config.label_size])

        # inference
        self.logits = self.inference()

        # loss
        if not self.config.is_training:
            return
        self.loss_val = self.loss()

        # train
        self.global_step = tf.Variable(0, trainable=False, name="Global_step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = self.config.decay_steps, self.config.decay_rate
        self.train_op = self.train_op()

        # accuracy
        # self.accuracy =



    # forward propagation
    def inference(self):
        with tf.variable_scope("embedding-layer"):
            embedding = tf.get_variable(name='embedding', shape=[self.config.vocab_size, self.config.embed_size])
        # transform word index to word vector
        hidden = tf.nn.embedding_lookup(embedding, self.sentence) # [None, sentence_len, embed_size]
        # average and squeeze
        self.hidden = tf.squeeze(tf.reduce_mean(hidden, axis=1, keepdims=True), axis=1) # [None, embed_size]


        with tf.variable_scope("linear-layer"):
            self.W = tf.get_variable(name="weights", shape=[self.config.embed_size, self.config.label_size], initializer=tf.truncated_normal_initializer(0,1))
            self.b = tf.get_variable(name='bias',shape=[self.config.label_size],initializer=tf.constant_initializer(0))
        logits = tf.matmul(self.hidden, self.W) + self.b  # [None, label_size]

        return logits

    def loss(self):
        if self.config.is_training:
            loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=tf.transpose(self.W),
                                  biases=self.b,
                                  labels=self.lables,
                                  inputs=self.hidden,
                                  num_sampled=self.config.num_sampled,
                                  num_true=self.config.max_label_per_example, # 这是与单标签的区别所在
                                  num_classes=self.config.label_size,
                                  partition_strategy='div'))

        else:
            labels_multi_hot = self.labels_l1999
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_multi_hot, logits=self.logits)
            loss = tf.reduce_sum(loss, axis=1)

        # add regularization result in not converge
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if "bias" not in v.name]) * self.config.l2_lambda
        loss += l2_loss
        return loss

    def train_op(self):
        self.learning_rate = tf.train.exponential_decay(self.config.learning_rate, self.global_step,
                                                   self.decay_steps, self.decay_rate, staircase=True, name='learning_rate')
        train_op = optimize_loss(loss=self.loss_val,
                                 global_step=self.global_step,
                                 learning_rate=self.learning_rate,
                                 optimizer="Adam")
        return train_op

# 测试训练过程
def test(config=Config()):
    fasttext = fastTextMulti()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(config.num_epochs):
            input_x = np.array([[1,2,5,0,3],[0,3,4,2,1]],dtype=np.int32) # [batch_size, sentence_len]
            input_y = np.array([[1,4,6,0,2],[0,1,2,3,4]], dtype=np.int32)
            input_y2 = np.array([[1,1,1,0,1,0,1,0,0,0],[1,1,1,1,1,0,0,0,0,0]],dtype=np.int32)
            loss,_ = sess.run([fasttext.loss_val, fasttext.train_op],
                              feed_dict={fasttext.sentence:input_x, fasttext.lables:input_y,
                                         fasttext.labels_l1999:input_y2})

            if i%config.print_very == 0:
                print("epoch:{0}, loss:{1}".format(i, loss))

if __name__ == "__main__":
    config = Config()
    test(config)


