# Hierarchical attention Networks for document classification.

print('started...')

import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.contrib.layers.python.layers import optimize_loss
import numpy as np
from tensorflow.contrib.layers.python.layers import xavier_initializer

class Config():
    def __init__(self):
        self.sequence_len = 30
        self.num_sentences = 6
        self.sentence_len = int(self.sequence_len / self.num_sentences)  # the number of words of each sencentenc
        self.num_classes = 3
        self.learning_rate = 0.01
        self.batch_size = 8
        self.decay_steps = 1000
        self.decay_rate = 0.9
        self.vocab_size = 10000
        self.embed_size = 100
        self.hidden_size = 100
        self.is_training = True
        # self.need_sentence_level_attention_encoder_flag = need_sentence_level_attention_encoder_flag
        self.multi_label_flag = False
        self.clip_gradients = 5.0
        self.cell_type = 'gru'
        self.dropout_keep_prob = 0.5
        self.initializer = xavier_initializer()
        self.l2_lambda = 0.5
  
class HAN():
    def __init__(self):
        self.config = Config()
        # add placeholder
        self.input_x = tf.placeholder(dtype=tf.int32, shape=[None, self.config.sequence_len], name='input_x')
        self.input_y = tf.placeholder(dtype=tf.int32, shape=[None], name='input_y')
        self.input_y_multilabel = tf.placeholder(tf.float32, [None, self.config.num_classes],name='input_y_multilabels')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # forward propagation
        self.initilization()
        self.logits = self.inference()

        self.predictions = tf.argmax(self.logits, 1, name='predictions')

        # computer loss
        self.loss = self.add_loss()

        # train
        self.global_step = tf.Variable(0, trainable=False, name='Global_step')
        self.epoch_step = tf.Variable(0, trainable=False, name='Epoch_step')
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = self.config.decay_steps, self.config.decay_rate
        self.train_op = self.train()

        # accuracy
        if not self.config.multi_label_flag:
            correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32), self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='Acurracy')
        else:
            self.accuracy = tf.constant(0.5)

    def initilization(self):
        with tf.variable_scope('embedding-layer'):
            self.embedding = tf.get_variable(name='embedding',shape=[self.config.vocab_size, self.config.embed_size],initializer=self.config.initializer)

        with tf.variable_scope('word-encoder'):
            # the forward direction and backward direction use the same weights!!!
            # the weights means the dynamic state transform metrix
            with tf.variable_scope('bi-rnn'):
                # the parameters of update
                self.Wz = tf.get_variable(name='Wz-e2h', shape=[self.config.embed_size, self.config.hidden_size], initializer=self.config.initializer)
                self.Uz = tf.get_variable(name='Uz-h2h', shape=[self.config.hidden_size, self.config.hidden_size], initializer=self.config.initializer)
                self.bz = tf.get_variable(name='bz', shape=[self.config.hidden_size], initializer=tf.constant_initializer(0.1))
                # the parameters of reset
                self.Wr = tf.get_variable(name='Wr-e2h', shape=[self.config.embed_size, self.config.hidden_size], initializer=self.config.initializer)
                self.Ur = tf.get_variable(name='Ur-h2h', shape=[self.config.hidden_size, self.config.hidden_size], initializer=self.config.initializer)
                self.br = tf.get_variable(name='br', shape=[self.config.hidden_size], initializer=tf.constant_initializer(0.1))
                # the parameters of new memory cell
                self.Wh = tf.get_variable(name='Wh-e2h', shape=[self.config.embed_size, self.config.hidden_size], initializer=self.config.initializer)
                self.Uh = tf.get_variable(name='Uh-h2h', shape=[self.config.hidden_size, self.config.hidden_size], initializer=self.config.initializer)
                self.bh = tf.get_variable(name='bh', shape=[self.config.hidden_size], initializer=tf.constant_initializer(0.1))

            with tf.variable_scope("word_attention"):
                self.Ww = tf.get_variable(name='Ww-attention-word', shape=[self.config.hidden_size*2, self.config.hidden_size*2], initializer=self.config.initializer)
                self.bw = tf.get_variable(name='bw-attention-word', shape=[self.config.hidden_size*2], initializer=tf.constant_initializer(0.1))
                # The word context vector uw is randomly initialized and jointly learned during the training process.
                self.uw = tf.get_variable(name='informative-word', shape=[self.config.hidden_size*2], initializer=self.config.initializer)

        with tf.variable_scope('sentence-encoder'):
            with tf.variable_scope('bi-rnn'):
                self.Wz_sentence = tf.get_variable(name='Wz-e2h-sentence', shape=[self.config.hidden_size*2, self.config.hidden_size*2], initializer=self.config.initializer)
                self.Uz_sentence = tf.get_variable(name='Uz-h2h-sentence', shape=[self.config.hidden_size*2, self.config.hidden_size*2], initializer=self.config.initializer)
                self.bz_sentence = tf.get_variable(name='bz-sentence', shape=[self.config.hidden_size*2], initializer=tf.constant_initializer(0.1))
                # the parameters of reset
                self.Wr_sentence = tf.get_variable(name='Wr-e2h-sentence', shape=[self.config.hidden_size*2, self.config.hidden_size*2], initializer=self.config.initializer)
                self.Ur_sentence = tf.get_variable(name='Ur-h2h-sentence', shape=[self.config.hidden_size*2, self.config.hidden_size*2], initializer=self.config.initializer)
                self.br_sentence = tf.get_variable(name='br-sentence', shape=[self.config.hidden_size*2], initializer=tf.constant_initializer(0.1))
                # the parameters of new memory cell
                self.Wh_sentence = tf.get_variable(name='Wh-e2h-sentence', shape=[self.config.hidden_size*2, self.config.hidden_size*2], initializer=self.config.initializer)
                self.Uh_sentence = tf.get_variable(name='Uh-h2h-sentence', shape=[self.config.hidden_size*2, self.config.hidden_size*2], initializer=self.config.initializer)
                self.bh_sentence = tf.get_variable(name='bh-sentence', shape=[self.config.hidden_size*2], initializer=tf.constant_initializer(0.1))

            with tf.variable_scope('sentence-attention'):
                self.Ws = tf.get_variable(name='Ws-attention-sentence', shape=[self.config.hidden_size*4, self.config.hidden_size*4], initializer=self.config.initializer)
                self.bs = tf.get_variable(name='bs-attention-sentence', shape=[self.config.hidden_size*4], initializer=tf.constant_initializer(0.1))
                # sentence level context vector can be randomly initialized and jointly learned during the training process.
                self.us = tf.get_variable(name='informative-sentence', shape=[self.config.hidden_size*4], initializer=self.config.initializer)

        with tf.variable_scope("Projection"):
            self.W_projection = tf.get_variable(name='weight_projection', shape=[self.config.hidden_size*4, self.config.num_classes], initializer=self.config.initializer)
            self.b_projection = tf.get_variable(name='bias-projection', shape=[self.config.num_classes],initializer=tf.constant_initializer(0))

    def inference(self):
        # word encoder -> word attention -> sentence encoder -> sentence attention -> linear classifier
        # 1. Word Encoder

        # 1.1 embedding of words
        input_x = tf.split(self.input_x, self.config.num_sentences, axis=1) # a list, have num_sentences elements. each element is [None, sentence_len = sequence_len/num_sentences]
        input_x = tf.stack(input_x, axis=1) # shape:[None, num_sentences, sentence_len]
        self.embeded_words = tf.nn.embedding_lookup(self.embedding, input_x) # [None, num_sentences, sentence_len, embed_size]
        embeded_words_reshaped = tf.reshape(self.embeded_words, shape=[-1, self.config.sentence_len, self.config.embed_size]) # [batch_size*num_sentences, sentence_len, embed_size]

        # split embeded_words to list
        embeded_words_splitted = tf.split(embeded_words_reshaped, self.config.sentence_len, axis=1)  # a list, length is sentence_len, each element is [batch_size*num_sentences, 1, embed_size]
        embeded_words_squeeze = [tf.squeeze(x, axis=1) for x in embeded_words_splitted]  # a list, length is sentence_len, each element is [batch_size*num_sentences, embed_size]

        # 1.2 forward gru
        hidden_state_forward_list = self.gru_word_level(embeded_words_squeeze) # return a list ,length is sentence_len, each element is [batch_size*num_sentences, hidden_size]

        # 1.3 backward gru
        embeded_words_squeeze.reverse()
        hidden_state_backward_list = self.gru_word_level(embeded_words_squeeze)
        hidden_state_backward_list.reverse() # reverse twice

        # 1.4 concat forward hidden state and backward hidden state.
        self.hidden_state_list = [tf.concat([h_forward, h_backward], axis=1) for h_forward, h_backward in
                             zip(hidden_state_forward_list,
                                 hidden_state_backward_list)]# zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。如果各个迭代器的元素个数不一致，则返回列表长度与最短的对象相同，利用 * 号操作符，可以将元组解压为列表。
        # return hidden_state: a list.len:sentence_length,element:[batch_size*num_sentences,hidden_size*2]


        # 2. word Attention
        sentence_representation = self.word_attention_level(self.hidden_state_list) # return: [batch_size*num_sentences, hidden_size*2]

        # 3. sentence encoder
        sentence_representation = tf.reshape(sentence_representation, [self.config.batch_size, self.config.num_sentences, self.config.hidden_size*2]) # [batch_size, num_sentences, hidden_size*2]

        # 3.1 gru forward
        sentence_representation_splitted = tf.split(sentence_representation, self.config.num_sentences, axis=1) # a list,length is num_sentences, each element is [self.config.batch_size,1,self.config.hidden_size*2]
        sentence_representation_squeeze = [tf.squeeze(x) for x in sentence_representation_splitted] # a list, each element is [self.config.batch_size, self.config.hidden_size*2]
        hidden_state_sentence_forward_list = self.gru_sentence_level(sentence_representation_squeeze) # a list, each element is [self.config.batch_size, self.config.hidden_size*2]

        # 3.2 gru backforward
        sentence_representation_squeeze.reverse()
        hidden_state_sentence_backward_list = self.gru_sentence_level(sentence_representation_squeeze)
        hidden_state_sentence_backward_list.reverse()

        # 3.4 concat forward hidden state and backward hidden state.
        self.hidden_state_sentence_list = [tf.concat([h_forward, h_backward], axis=1) for h_forward, h_backward in
                                      zip(hidden_state_sentence_forward_list, hidden_state_sentence_backward_list)]
        # a list, each element is [self.config.batch_size, self.config.hidden_size*4]

        # 4. sentence attention
        document_representation = self.sentence_attention_level(self.hidden_state_sentence_list) # [batch_size, hidden_size*4]

        # 5. dropout
        with tf.name_scope('dropout'):
            h_dropout = tf.nn.dropout(document_representation, keep_prob=self.config.dropout_keep_prob, name='dropout')

        # 5. logits(use linear layer)and predictions(argmax)
        logits = tf.matmul(h_dropout, self.W_projection) + self.b_projection # [batch_size, num_classes]
        return logits

    def add_loss(self):
        if not self.config.multi_label_flag:
            print("Going to use single label loss.")
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
        else:
            print("Going to use multi label loss.")
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_multilabel, logits=self.logits)
        loss = tf.reduce_mean(losses)  # print("2.loss.loss:", loss) #shape=()
        l2_losses = tf.add_n(
            [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * self.config.l2_lambda
        loss = loss + l2_losses
        return loss

    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.config.learning_rate, self.global_step, self.decay_steps,
                                                   self.decay_rate, staircase=True)
        # noise_std_dev = tf.constant(0.3) / (tf.sqrt(tf.cast(tf.constant(1) + self.global_step, tf.float32))) #gradient_noise_scale=noise_std_dev
        train_op = optimize_loss(self.loss, global_step=self.global_step,
                                                   learning_rate=learning_rate, optimizer="Adam",
                                                   clip_gradients=self.config.clip_gradients)
        return train_op

    # one step of gru
    def gru_single_step_word_level(self, Xt, h_t_prev):
        """
        single step of gru for word level
        :param Xt: embedding of the current word   [batch_size*num_sentences, embed_size]
        :param h_t_prev: the hidden state of the previous state   [batch_size*num_sentences, hidden_size]
        :return: h_t: [batch_size*num_sentences,hidden_size]
        """
        # the reset gate
        r_t = tf.nn.sigmoid(tf.matmul(Xt, self.Wr) + tf.matmul(h_t_prev, self.Ur) + self.br) # [batch_size*num_sentences, hidden_size]
        # the new memory cell
        h_t_candidate = tf.nn.tanh(tf.matmul(Xt, self.Wh) + tf.multiply(r_t, tf.matmul(h_t_prev, self.Uh)) + self.bh)  # [batch_size*num_sentences, hidden_size]
        # the update gate
        z_t = tf.nn.sigmoid(tf.matmul(h_t_prev, self.Wz) + tf.matmul(h_t_prev, self.Uz) + self.bz) # [batch_size*num_sentences, hidden_size]
        # the new state
        h_t = tf.multiply(h_t_prev, 1 - z_t) + tf.multiply(z_t, h_t_candidate) # [batch_size*num_sentences, hidden_size]
        return h_t

    # gru
    def gru_word_level(self, embeded_words_squeeze):
        """
        :param embeded_words: a list, length is sentence_len, each element is [batch_size*num_sentences, embed_size]
        :return: a list.length is sentence_length, each element is [batch_size*num_sentences,hidden_size]
        """
        # initilization hidden state
        h_t = tf.ones(shape=[self.config.batch_size * self.config.num_sentences, self.config.hidden_size])
        h_t_list = []
        for time_step, Xt in enumerate(embeded_words_squeeze):
            h_t = self.gru_single_step_word_level(Xt, h_t) # update the previous hidden state directly
            h_t_list.append(h_t)
        return h_t_list

    def word_attention_level(self, hidden_state_list):
        """
        :param hidden_state_list: a list, sentence_len elements, and each element is [batch_size*num_sentences, hidden_size*2]
        :return:
        """
        # tf.stack()
        # Given a list of length `N` of tensors of shape `(A, B, C)`;
        # if `axis == 0` then the `output` tensor will have the shape `(N, A, B, C)`.
        # if `axis == 1` then the `output` tensor will have the shape `(A, N, B, C)`.
        hidden_state_1 = tf.stack(hidden_state_list, axis=1)  # [batch_size*num_sentences, sentence_len, hidden_size*2]

        # 为了方便所有的hidden_state与矩阵model.config.Ww相乘，将其shape转换为 [batch_size*num_sentences*sentence_len, hidden_size*2]
        hidden_state_2 = tf.reshape(hidden_state_1,
                                    [-1,
                                     self.config.hidden_size * 2])  # [batch_size*num_sentences*sentence_len, hidden_size*2]

        # 1) one-layer MLP: hidden_representation is the u_{it}
        u_it = tf.nn.tanh(tf.matmul(hidden_state_2, self.Ww) + self.bw,
                          name='hidden_representation')  # [batch_size*num_sentences*sentence_len, hidden_size*2]
        u_it = tf.reshape(u_it, [-1, self.config.sentence_len,
                                 self.config.hidden_size * 2])  # [batch_size*num_sentences, sentence_len, hidden_size*2]

        # 2) get the importance of the word as the similarity of u_{it} with a word level context vector u_w

        # model.config.uw是共享的，并且是需要训练的参数。与所有的 hidden state 矩阵相乘，求得相似度
        # u_it.shape=[batch_size*num_sentences, sentence_len, hidden_size*2]  model.config.uw.shape=[model.config.hidden_size*2]
        similarity_with_uit_uw = tf.reduce_sum(tf.multiply(u_it, self.uw), axis=2,
                                               keepdims=False)  # axis=2 is a scale, [batch_size*num_sentence, sentence_len]
        # subtract the maximum, avoid the Numerical overflow
        similarity_with_uit_uw_max = tf.reduce_max(similarity_with_uit_uw, axis=1,
                                                   keepdims=True)  # [batch_size*num_sentence, 1]

        # get possibility distribution for each word in the sentence, a normalized importance weight α_it through a softmax function
        alpha_it = tf.nn.softmax(similarity_with_uit_uw - similarity_with_uit_uw_max,
                                 name='word_attention')  # [batch_size*num_sentence, sentence_len]

        # 3) sentence vector s_i
        alpha_it_expended = tf.expand_dims(alpha_it, axis=2)  # [batch_size*num_sentence, sentence_len, 1]
        sentence_representation = tf.multiply(alpha_it_expended,
                                              hidden_state_1)  # [batch_size*num_sentence, sentence_len, hidden_size*2]
        # 将一个sentence中所有的经过attention加权了权重的word的向量表示累加，得到sentence的向量表示
        sentence_representation = tf.reduce_sum(sentence_representation, axis=1,
                                                keepdims=False)  # [batch_size*num_sentence, hidden_size*2]
        return sentence_representation

    def gru_single_step_sentence_level(self, Xt, h_t_prev):
        """
        :param Xt: one sentence presentation of the document [batch_size, hidden_size*2]
        :param h_t_prev: the previous sentence hidden state in the document [hidden_size*2, hidden_size*2]
        :return:
        """
        # reset gate
        r_t = tf.nn.sigmoid(tf.matmul(Xt, self.Wr_sentence) + tf.matmul(h_t_prev, self.Ur_sentence)+ self.br_sentence)  # [batch_size, hidden_size*2]
        # candidate state
        h_t_candidate = tf.nn.tanh(tf.matmul(Xt, self.Wh_sentence) + tf.multiply(r_t, tf.matmul(h_t_prev, self.Uh_sentence)) + self.bh_sentence) # [batch_size, hidden_size*2]
        # update gate
        z_t = tf.nn.sigmoid(tf.matmul(Xt, self.Wz_sentence) + tf.matmul(h_t_prev, self.Uz_sentence) + self.bz_sentence)  # [batch_size, hidden_size*2]
        # the new hidden state
        h_t = tf.multiply(z_t, h_t_candidate) + tf.multiply(1- z_t, h_t_prev)
        return h_t

    def gru_sentence_level(self, sentence_presentation_squeeze):
        """
        :param sentence_presentation: # a list, each element is [self.config.batch_size, self.config.hidden_size*2]
        :return: # a list, each element is [self.config.batch_size, self.config.hidden_size*2]
        """
        # the init sentence state
        h_t = tf.ones(shape=[self.config.batch_size, self.config.hidden_size*2])
        h_t_list = []
        for time_step, hidden_state_sen in enumerate(sentence_presentation_squeeze):
            h_t = self.gru_single_step_sentence_level(hidden_state_sen, h_t)
            h_t_list.append(h_t)
        return h_t_list

    def sentence_attention_level(self, hidden_state_sentence_list):
        """
        :param hidden_state_sentence_list: a list, length is num_sentence, each element is [bath_size, hidden_size*4]
        :return:
        """
        hidden_state_sentence_1 = tf.stack(hidden_state_sentence_list,
                                           axis=1)  # [bath_size, num_sentences, hidden_size*4]
        # attention 是可以批量处理的，
        hidden_state_sentence_2 = tf.reshape(hidden_state_sentence_1,
                                             [-1,
                                              self.config.hidden_size * 4])  # [bath_size*num_sentences, hidden_size*4]

        # one MLP layer, hidden_sentence_representation is the u_{i}
        u_i = tf.nn.tanh(tf.matmul(hidden_state_sentence_2, self.Ws) + self.bs,
                         name='hidden_sentence_representation')  # [bath_size*num_sentences, hidden_size*4]
        u_i = tf.reshape(u_i, [self.config.batch_size, self.config.num_sentences,
                               self.config.hidden_size * 4])  # [bath_size, num_sentences, hidden_size*4]

        # the similarity
        similarity_ui_uw = tf.reduce_sum(tf.multiply(u_i, self.us), axis=2,
                                         keepdims=False)  # [batch_size, num_sentences]
        similarity_ui_uw_max = tf.reduce_max(similarity_ui_uw, axis=1, keepdims=True)  # [bath_size, 1]

        alpha_i = tf.nn.softmax(similarity_ui_uw - similarity_ui_uw_max)  # [batch_size, num_sentences]
        alpha_i_expanded = tf.expand_dims(alpha_i, axis=2)  # [batch_size, num_sentences]

        # the document representation
        document_representation = tf.multiply(alpha_i_expanded,
                                              hidden_state_sentence_1)  # [bath_size, num_sentences, hidden_size*2]
        # 将一个document中所有的经过attention加权了权重的sentence的向量表示累加，得到document的向量表示
        document_representation = tf.reduce_sum(document_representation, axis=1,
                                                keepdims=False)  # [bath_size, hidden_size*2]
        return document_representation


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

if __name__ == "__main__":
    config = Config()
    han = HAN()
    input_x = np.ones([config.batch_size, config.sequence_len])
    input_y = np.array([1, 0, 1, 1, 1, 2, 1, 1])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(100):
            loss, accuracy, _ = sess.run([han.loss, han.accuracy, han.train_op],
                                         feed_dict={han.input_x:input_x, han.input_y:input_y})
            if epoch%8 == 0:
                print("loss:{0}, accuracy:{1}".format(loss, accuracy))

