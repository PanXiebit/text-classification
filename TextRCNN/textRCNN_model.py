print("started...")

import tensorflow as tf
import gensim
import numpy as np
from tensorflow.contrib.layers.python.layers import optimize_loss


class RCNN:
    def __init__(self, sequence_len, batch_size, embed_size, vocab_size, state_size,hidden_size, num_classes, learning_rate,
                 decay_steps, decay_rate, num_iter, is_training=True,cell_type = None, multi_label_flag=False,
                 dropout_keep_prob=0.5, clip_gradients=0.5, l2_lambda=0.5, is_pretrained=False):
        self.sequence_len = sequence_len
        self.batch_size = batch_size
        self.embed_szie = embed_size
        self.vocab_size = vocab_size
        self.state_size = state_size   # rnn中隐藏状态的维度
        self.hidden_size = hidden_size # y^{(2)} 对应每个词的维度
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.num_iter = num_iter
        self.is_training = is_training
        self.multi_label_flag = multi_label_flag
        self.clip_gradients = clip_gradients
        self.is_pretrained = is_pretrained
        self.cell_type = cell_type
        self.dropout_keep_prob = dropout_keep_prob
        self.l2_lambda  = l2_lambda

        # add placeholder
        self.input_x = tf.placeholder(dtype=tf.int32, shape=[None, self.sequence_len], name='input_x')
        self.input_y = tf.placeholder(dtype=tf.int32, shape=[None], name='input_y')
        self.input_y_multilabels = tf.placeholder(dtype=tf.int32, shape=[None, self.num_classes])

        # forward propagation
        self.logits = self.inference()
        self.predict = tf.argmax(self.logits, axis=1, name='prediction')

        self.loss = self.add_loss()

        # train
        self.global_steps = tf.Variable(0, trainable=False, name='global_steps')
        self.epoch_steps = tf.Variable(0, trainable=False, name='train_steps')
        self.epoch_increment = tf.assign(self.epoch_steps, tf.add(tf.constant(1), self.epoch_steps))
        self.train_op = self.add_train_op()



    def inference(self):
        # embedding
        with tf.variable_scope("embedding-layer"):
            if not self.is_pretrained:
                embedding = tf.get_variable(name='embedding', shape=[self.vocab_size, self.embed_szie], initializer=tf.truncated_normal_initializer(0))
            else: # 还未下载词向量
                word2vec = gensim.models.Word2Vec.load('word2vec.gensim')
                embedding = np.zeros((word2vec.syn0_lockf.shape[0]+1, word2vec.syn0.shape[1]), dtype=np.float32)
            self.embedd_words = tf.nn.embedding_lookup(embedding, self.input_x) # [None, sequence_len, embed_size]

        # bi-rnn
        with tf.variable_scope('bi-rnn'):
            cell_left_init = tf.get_variable('_init',shape=[self.batch_size, self.state_size], initializer=tf.truncated_normal_initializer(0))
            cell_right_init = tf.get_variable('w_right_init', shape=[self.batch_size, self.state_size],
                                        initializer=tf.truncated_normal_initializer(0))
            # W_l = tf.get_variable('wl_h2h', shape=[self.state_size, self.state_size],initializer=tf.truncated_normal_initializer(0))
            # W_l_e2h = tf.get_variable('wl_embed2h', shape=[self.embed_szie, self.state_size],initializer=tf.truncated_normal_initializer(0))
            # w_r = tf.get_variable('w_right_h2h', shape=[self.state_size, self.embed_szie], initializer=tf.truncated_normal_initializer(0))
            # w_r_e2h = tf.get_variable('wr_embed2h', shape=[self.embed_szie, self.state_size], initializer=tf.truncated_normal_initializer(0))

            # forward direction
            fw_cell = self._get_cell(self.state_size, self.cell_type) # BasicRNNCell object
            # dropout ??
            fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=self.dropout_keep_prob)
            # backword direction
            bw_cell = self._get_cell(self.state_size, self.cell_type)
            bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=self.dropout_keep_prob)

            # If time_major == False (default),
            # output_fw will be a `Tensor` shaped: `[batch_size, max_time, cell_fw.output_size]`
            # and output_bw will be a `Tensor` shaped: `[batch_size, max_time, cell_bw.output_size]` # max_time=sequence_len
            # A tuple (output_state_fw, output_state_bw) containing the forward and the backward final states of bidirectional rnn.
            (self.output_fw, self.output_bw), states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                                                       cell_bw=bw_cell,
                                                                                       inputs=self.embedd_words,
                                                                                       initial_state_fw=cell_left_init,
                                                                                       initial_state_bw=cell_right_init)


        with tf.name_scope("context"):
            # _dynamic_rnn 的输出是 outputs, state,
            # 其中 outputs 是每一个时间步的隐藏状态，shape [batch_size, max_time, cell_state_size]
            # states 是最后一个时间步的输出 shape [batch_size, cell_state_size]
            # 所以这里的cell_fw.output_size = cell_bw.output_size = cell_state_size = state_size
            shape = [tf.shape(self.output_fw)[0], 1, tf.shape(self.output_fw)[2]] # [batch_size, 1, cell_fw.output_size]
            self.c_left = tf.concat([tf.zeros(shape), self.output_fw[:, :-1]], axis=1, name='context_left') # [bath_size, 1+ (max_time-1), cell_fw.output_size]
            self.c_right = tf.concat([self.output_bw[:, 1:],tf.zeros(shape)], axis=1, name='context_right') #

        with tf.name_scope('word-representation'):
            self.x = tf.concat([self.c_left, self.embedd_words, self.c_right], axis=2, name='X') # [batch_size,max_time, 2*output_size + embed_size]
            embedding_size = 2 * self.state_size + self.embed_szie

        # 这一步也可以看做卷积操作，只不过filter_size=1,过滤器W2分别与self.x中的每一个词的向量表示做卷积。
        with tf.variable_scope('text-representation'):
            W2 = tf.get_variable(name='W2', shape=[embedding_size, self.hidden_size], initializer=tf.random_uniform_initializer(-1,1))
            b2 = tf.get_variable(name='b2', shape=[self.hidden_size],initializer=tf.constant_initializer(0))
            # def einsum(equation, *inputs, **kwargs)  equation必须要写，可以看源代码有哪些形式
            # Batch matrix multiplication:
            # einsum('aij,jk->aik', s, t)  # out[a,i,k] = sum_j s[a,i,j] * t[j, k]
            self.y2 = tf.einsum('aij,jk->aik',self.x, W2) + b2 # [batch_size, max_time, hidden_size]

        with tf.name_scope('max-pooling'):
            self.y3 = tf.reduce_max(self.y2, axis=1) #[batch_size, hidden_size]

        with tf.variable_scope('output'):
            W4 = tf.get_variable('W4', [self.hidden_size,self.num_classes],initializer=tf.truncated_normal_initializer(0)) #[batch_size, num_classes]
            b4 = tf.get_variable('b4', [self.num_classes], initializer=tf.constant_initializer(0.1))
            logits = tf.nn.xw_plus_b(self.y3, W4, b4, name='logits')
            return logits

    # computer loss
    def add_loss(self):
        if not self.multi_label_flag:
            self.input_y_onehot = tf.one_hot(indices=self.input_y, depth=self.num_classes)
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y_onehot)
        else:
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y_multilabels)
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * self.l2_lambda
        loss = tf.reduce_mean(losses) + l2_loss
        return loss

    @staticmethod
    def _get_cell(state_size, cell_type):
        if cell_type == 'vanilla':
            return tf.nn.rnn_cell.BasicRNNCell(state_size)
        if cell_type == 'lstm':
            return tf.nn.rnn_cell.BasicLSTMCell(state_size)
        if cell_type == 'gru':
            return tf.nn.rnn_cell.GRUCell(state_size)
        else:
            print("ERROR: '" + cell_type + "' is a wrong cell type")
            return None

    # train operation
    def add_train_op(self):
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_steps, self.decay_steps, self.decay_rate)
        train_op1 = optimize_loss(self.loss,self.global_steps,learning_rate,optimizer='Adam',clip_gradients=self.clip_gradients)
        train_op2 = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, self.global_steps)
        return train_op2


# test started
def test():
    num_classes = 10
    learning_rate = 0.01
    batch_size = 8
    sequence_len = 5
    embed_size = 100
    vocab_size = 1000
    state_size = 100
    hidden_size = 50
    decay_steps = 100
    decay_rate = 0.9
    num_iter = 1000
    cell_type = 'vanilla'
    text_rcnn = RCNN(sequence_len, batch_size, embed_size, vocab_size, state_size, hidden_size, num_classes, learning_rate,
                 decay_steps, decay_rate, num_iter, cell_type=cell_type)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(num_iter):
            input_x = np.zeros(shape=[batch_size, sequence_len])
            input_y = np.array([1,0,2,3,1,3,1,0])
            loss, _ = sess.run([text_rcnn.loss, text_rcnn.train_op],
                               feed_dict={text_rcnn.input_x:input_x, text_rcnn.input_y: input_y})
            print('{0} epoch,loss:{1}'.format(i, loss))

if __name__ == "__main__":
    test()
