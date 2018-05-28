print('started...')
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import optimize_loss



class Config():
    def __init__(self):
        self.label_size = 20  # 文档分类的总类别数
        self.batch_size = 8
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
        if self.config.is_training:
            return
        self.loss = self.loss()

        # train
        self.global_step = tf.Variable(0, trainable=False, name="Global_step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = self.config.decay_steps, self.config.decay_rate
        self.train_op = self.train_op()


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
            loss = tf.nn.nce_loss(weights=tf.transpose(self.W),
                                  biases=self.b,
                                  labels=self.lables,
                                  inputs=self.hidden,
                                  num_sampled=self.config.num_sampled,
                                  num_true=self.config.max_label_per_example, # 这是与单标签的区别所在
                                  num_classes=self.config.label_size,
                                  partition_strategy='div')

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
        train_op = optimize_loss(self.loss, self.global_step, self.learning_rate, optimizer='Adam')

        return train_op


if __name__ == "__main__":
    config = Config()
    input_x = tf.random_uniform(shape=[config.batch_size, config.sentence_len])
    input_y = tf.zeros(shape=[config.batch_size, config.max_label_per_example])

