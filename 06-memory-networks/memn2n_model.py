# implement of end to end memory networks

import tensorflow as tf
import numpy as np
import os
from tensorflow.contrib.layers.python.layers import optimize_loss


def position_encoding(sentence_size, embedding_size):
    """
    Position Encoding described in section 4.1 [1]
    """
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    le = embedding_size+1
    ls = sentence_size + 1
    for k in range(1, le):
        for j in range(1, ls):
            # here is different from the paper.
            # the formulation in paper is: l_{kj}=(1-j/J)-(k/d)(1-2j/J)
            # here the formulation is: l_{kj} = 1+4(k- (d+1)/2)(j-(J+1)/2)/d/J,
            # 具体表现可查看 https://www.wolframalpha.com/input/?i=1+%2B+4+*+((y+-+(20+%2B+1)+%2F+2)+*+(x+-+(50+%2B+1)+%2F+2))+%2F+(20+*+50)+for+0+%3C+x+%3C+50+and+0+%3C+y+%3C+20
            encoding[k-1, j-1] = (k - (embedding_size+1)/2) * (j - (sentence_size+1)/2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    # Make position encoding of time words identity to avoid modifying them
    encoding[:, -1] = 1.0 # 最后一个sentence的权重都为1
    return np.transpose(encoding) # [sentence_size, embedding_size]

def zeros_nil_slot(t, name=None):
    """
    The nil_slot is a dummy slot and should not be trained and influence
    the training algorithm.
    """
    with tf.op_scope([t], name, 'zero_nil_slot') as name:
        t = tf.convert_to_tensor(t, name='t')
        s = tf.shape(t)[1]
        z = tf.zeros(tf.stack([1, s]))
        return tf.concat(axis=0, values=[z, tf.slice(t,[1,0],[-1, -1])], name=name)

def add_gradient_noise(t, stddev=1e-3, name=None):
    """
    Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].
    The input Tensor `t` should be a gradient.
    The output will be `t` + gaussian noise.
    0.001 was said to be a good fixed value for memory networks [2].
    """
    with tf.op_scope([t, stddev], name, "add_gradient_noise") as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)

class E2EMemm(object):
    """end to end memory networks"""
    def __init__(self, batch_size, sentence_len, memory_size, vocab_size, embed_size,
                 hops=3,
                 max_grad_norm=40.0,
                 nonlin=None,
                 initializer=tf.random_normal_initializer(mean=0,stddev=0.1),
                 encoding=position_encoding,
                 is_training = True,
                 name='MemN2N'):
        """Creats an end-to-end memory networks

        :param batch_size:
        :param sentence_len:
        :param memory_size: the number of sentences in a single story.
        :param vocab_size: The size of the vocabulary (should include the nil word). The nil word
            one-hot encoding should be 0.
        :param embed_size: The max size of a sentence in the data. All sentences should be padded
            to this length. If padding is required it should be done with nil one-hot encoding (0).
        :param hops: The number of hops. A hop consists of reading and addressing a memory slot.
            Defaults to `3`.
        :param max_grad_norm: Maximum L2 norm clipping value. Defaults to `40.0`.
        :param nonlin: Non-linearity. Defaults to `None`.
        :param initializer: Weight initializer. Defaults to `tf.random_normal_initializer(stddev=0.1)`
        :param encoding: A function returning a 2D Tensor (sentence_size, embedding_size). Defaults to `position_encoding`.
        :param sess:
        :param name:
        """
        self._batch_size = batch_size
        self._sentence_len = sentence_len
        self._memroy_size = memory_size
        self._vocab_szie = vocab_size
        self._embed_szie = embed_size
        self._hops = hops
        self._max_grad_norm = max_grad_norm
        self._nonlin = nonlin
        self._init = initializer
        self._name = name
        self._is_training = is_training

        # 1. add placeholder
        with tf.name_scope("add_placeholder"):
            self._stories = tf.placeholder(tf.int32, [None, self._memroy_size, self._sentence_len], name="input_stories") # 这里的memory_size就是一个story中含有多少个sentence
            self._queries = tf.placeholder(tf.int32, [None, self._sentence_len], name='query')
            self._answer = tf.placeholder(tf.int32, [None, self._vocab_szie], name='answer') # 单个词，输出one-hot向量
            self._lr = tf.placeholder(tf.float32, [], name="learning-rate")
            tf.summary.scalar('learning_rate', self._lr)

        # 2. init embedding matrix A,B,C
        with tf.name_scope("init_embedding_matrix"):
            self._build_vars()
            # accoding to the word order, get the weighted matrix [sentence_len, embed_size]
            self._encoding = tf.constant(encoding(self._sentence_len, self._embed_szie), name='encoding') # 一个特殊的矩阵，根据词序计算得到的权重矩阵
            tf.summary.histogram('position-encoding', self._encoding)

        # 3. computer logits
        with tf.name_scope('predicts'):
            self._logits = self._inference()  # [None, vocab_size]
            self.predict = tf.argmax(self._logits, axis=1, name='prediction')

        # 4. accuracy
        with tf.name_scope('accuracy'):
            correct_prediction = tf.cast(tf.equal(self.predict, tf.argmax(self._answer, 1)), tf.int32)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='Accuracy')
            tf.summary.scalar('accuracy', self.accuracy)
            if not self._is_training:
                return

        # 5. add loss
        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self._answer, logits= self._logits, name='cross_entropy')
            # 是否需要添加正则化??
            self._loss = tf.reduce_sum(cross_entropy, name='loss')
            tf.summary.scalar('loss', self._loss)

        # 6. train operation
        with tf.name_scope('train'):
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.epoch_step = tf.Variable(0, trainable=False, name='epoch_step')
            epoch_step = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
            self.train_op = self._train()

    # padding if needed
    def _build_vars(self):
        with tf.variable_scope(self._name):
            nil_word_slot = tf.zeros([1, self._embed_szie])  # nil word
            # initilizate memory embedding matrix
            A = tf.concat(axis=0, values=[nil_word_slot, self._init([self._vocab_szie-1, self._embed_szie])], name='memory-embedding-matrix')
            # initilizate output embedding matrix
            C = tf.concat(axis=0, values=[nil_word_slot, self._init([self._vocab_szie-1, self._embed_szie])], name='output-embedding-matrix')

            self.A_1 = tf.Variable(A, name='A') # A_1为初始设置，之后的A^{k+1}=C^k
            tf.summary.histogram('embedding-matrix-A', self.A_1)
            self.C = []

            for hop in range(self._hops):
                with tf.variable_scope('hop_{}'.format(hop+1)):
                    self.C.append(tf.Variable(C, name='C_{}'.format(hop+1)))
        self._nil_vars = set([self.A_1.name] + [x.name for x in self.C]) #

    def _inference(self):
        # accoding to the Adjacent weight sharing, question embedding matrix B=A_1
        q_embed = tf.nn.embedding_lookup(self.A_1, self._queries) # [None, sentence_len, embed_size]
        #  internal state u:点乘根据词序得到的权重矩阵 u_0 = \sum_jl_j\cdot Bx_{ij}
        u_k = tf.reduce_sum(q_embed * self._encoding, axis=1) # [None, embed_size]

        for hop in range(self._hops):
            if hop == 0:
                # memory representation
                m_embed_A = tf.nn.embedding_lookup(self.A_1, self._stories) # [None, memory_size, sentence_len, embed_size]
                # weighted with the encoding matrix
                m_embed_A = tf.reduce_sum(m_embed_A * self._encoding, axis=2) # [None, memory_size, embed_size]

            else:
                with tf.variable_scope('hop_{}'.format(hop-1)):
                    # A^{k+1} = C^k
                    m_embed_A = tf.nn.embedding_lookup(self.C[hop-1], self._stories)
                    m_embed_A = tf.reduce_sum(m_embed_A * self._encoding, axis=2) #[None, memory_size, embed_size]

            # computer the similarity of the memory and query
            u_temp = tf.transpose(tf.expand_dims(u_k, -1), [0,2,1]) # [None, embed_size, 1] -> [None, 1, embed_size]
            similarity = tf.reduce_sum(m_embed_A * u_temp, axis=2) # [None, memory_size], broadcast

            # caculate probabilities
            probs = tf.nn.softmax(similarity) # [None, memory_size]
            # probs_temp = tf.transpose(tf.expand_dims(probs, -1), [0,2,1]) # [None, 1, memory_size]
            probs_temp = tf.expand_dims(probs, axis=1)  # [None, 1, memory_size]

            # caculate the output: o=\sum_i p_ic_i
            with tf.variable_scope('hop_{}'.format(hop)):
                m_embed_C = tf.nn.embedding_lookup(self.C[hop], self._stories) # [None, memory_size, sentence_len, embed_size]
            # multipy the same weighted matrix
            m_embed_C = tf.reduce_sum(m_embed_C * self._encoding, axis=2) # [None, memory_size, embed_size]
            m_embed_C_T= tf.transpose(m_embed_C, [0, 2, 1]) # [None, embed_size, memory_size]

            # context vector
            output = tf.reduce_sum(m_embed_C_T * probs_temp, axis=2) # [None, embed_size]

            # u_{k+1} = o_k + u_k
            u_k= output + u_k      # [None, embed_size]

            if self._nonlin:
                u_k = self._nonlin(u_k)

            with tf.variable_scope('hop_{}'.format(self._hops)):
                # W^T = C^k  [embed_size, vocab_size]
                return tf.matmul(u_k, tf.transpose(self.C[-1]))  # [None, vocab_size]

    def _train(self):
        # train_op = optimize_loss(self._loss, global_step=self.global_step,learning_rate=self._lr,
        #                          optimizer='SGD', clip_gradients=self._max_grad_norm)
        # 执行梯度下降的另一种方式
        self._opt = tf.train.GradientDescentOptimizer(self._lr)
        grads_and_vars = self._opt.compute_gradients(self._loss)
        grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v) for g,v in grads_and_vars]
        grads_and_vars = [(add_gradient_noise(g), v) for g,v in grads_and_vars]
        nil_grads_abd_vars = []
        for g,v in grads_and_vars:
            if v.name in self._nil_vars:
                nil_grads_abd_vars.append((zeros_nil_slot(g), v))
            else:
                nil_grads_abd_vars.append((g, v))
        train_op = self._opt.apply_gradients(nil_grads_abd_vars, name='train_op')
        return train_op

    # def predict(self, stories, queries):
    #     """predicts answer as one-hot encoding.
    #
    #     :param stories: Tensor (None, memory_size, sentence_size)
    #     :param queries: Tensor (None, sentence_size)
    #     :return: answer: Tensor (None. vocab_size)
    #     """
    #     feed_dict = {self._stories: stories, self._queries:queries}
    #     return self.



if __name__ == '__main__':
    # 一个小的test
    stories = np.random.randint(0, 100, size=(8, 10, 10))
    queries = np.random.randint(0,100,(8, 10))
    answer = tf.one_hot([1,4,5,2,7,5,4,55], depth=100)
    model = E2EMemm(batch_size=8, memory_size=10, sentence_len=10, vocab_size=100, embed_size=10)
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter('./tmp/summary/train', sess.graph)
        sess.run(tf.global_variables_initializer())

        # # 查看以下哪些变量是可训练的，哪些是不可训练的
        # print(tf.GraphKeys.GLOBAL_VARIABLES) # http://www.panxiaoxie.cn/2018/05/16/tensorflow-API-Building-Graphs/
        # print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        # print(tf.GraphKeys.TRAINABLE_VARIABLES)  # trainable_variables
        # print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

        answer = answer.eval() # feed 不能为 tensor
        for i in range(100):
            feed_dict = {model._stories: stories, model._queries:queries, model._answer:answer, model._lr:0.01}
            summary,loss,  _ = sess.run([merged, model._loss, model.train_op], feed_dict)
            if i%10 == 0:
                print("{} epoch, loss:{}".format(i, loss))
            train_writer.add_summary(summary, i) # Adds a `Summary` protocol buffer to the event file.
        train_writer.close()
