# Hierarchical attention Networks for document classification.

print('started...')

import tensorflow as tf

class HAN:
    def __init__(self, sequence_len, num_sentences, num_classes, learning_rate, batch_size, decay_steps, decay_rate,
                 vocab_size, embed_size, hidden_size, is_training, need_sentence_level_attention_encoder_flag=True,
                 multi_label_flag=False, clip_gradients=5.0, cell_type='gru'):

        self.sequence_len = sequence_len
        self.num_sentences = num_sentences
        self.sentence_len = sequence_len / num_sentences  # the number of words of each sencentenc
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.is_training = is_training
        self.need_sentence_level_attention_encoder_flag = need_sentence_level_attention_encoder_flag
        self.multi_label_flag = multi_label_flag
        self.clip_gradients = clip_gradients
        self.cell_type = cell_type


        # add placeholder
        self.input_x = tf.placeholder(dtype=tf.int32, shape=[None, self.sequence_len], name='input_x')
        self.input_y = tf.placeholder(dtype=tf.int32, shape=[None], name='input_y')
        self.input_y_multilabel = tf.placeholder(tf.float32, [None, self.num_classes],name='input_y_multilabels')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        self.global_step = tf.Variable(0, trainable=False, name='Global_step')
        self.epoch_step = tf.Variable(0, trainable=False, name='Epoch_step')
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.logits = self.inference()

        self.predictions = tf.argmax(self.logits, 1, name='predictions')

        if not self.multi_label_flag:
            correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32), self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='Acurracy')
        else:
            self.accuracy = tf.constant(0.5)

        # forward propagation
        self.logits = self.inference()

        if not is_training:
            return
        if multi_label_flag:
            print("Going to use multi label loss.")
            self.loss = self.add_loss_multilabel()
        else:
            print("Going to use single label loss.")
            self.loss = self.add_loss()

    def initilization(self):
        with tf.Variable('embedding-layer'):
            self.embedding = tf.get_variable(name='embedding',shape=[self.vocab_size, self.embed_size],initializer=tf.truncated_normal_initializer(0))

        with tf.variable_scope('word-encoder'):
            with tf.variable_scope('bi-rnn'):
                # the parameters of update
                self.Wz = tf.get_variable(name='weights-e2h-Wz', shape=[self.embed_size, self.hidden_size], initializer=tf.truncated_normal_initializer(0))
                self.Uz = tf.get_variable(name='weights-h2h-Uz', shape=[self.hidden_size, self.hidden_size], initializer=tf.truncated_normal_initializer(0))
                self.bz = tf.get_variable(name='bias-bz', shape=[self.hidden_size], initializer=tf.constant_initializer(0.1))
                # the parameters of reset
                self.Wr = tf.get_variable(name='weights-e2h-Wr', shape=[self.embed_size, self.hidden_size], initializer=tf.truncated_normal_initializer(0))
                self.Ur = tf.get_variable(name='weights-h2h-Ur', shape=[self.hidden_size, self.hidden_size], initializer=tf.truncated_normal_initializer(0))
                self.br = tf.get_variable(name='bias-br', shape=[self.hidden_size], initializer=tf.constant_initializer(0.1))
                # the parameters of new memory cell
                self.Wh = tf.get_variable(name='weights-e2h-Wh', shape=[self.embed_size, self.hidden_size], initializer=tf.truncated_normal_initializer(0))
                self.Uh = tf.get_variable(name='weights-h2h-Uh', shape=[self.hidden_size, self.hidden_size], initializer=tf.truncated_normal_initializer(0))
                self.bh = tf.get_variable(name='bias-bh', shape=[self.hidden_size], initializer=tf.constant_initializer(0.1))


    def inference(self):
        # word encoder -> word attention -> sentence encoder -> sentence attention -> linear classifier
        # 1. Word Encoder
        # 1.1 embedding of words
        input_x = tf.split(self.input_x, self.num_sentences, axis=1) # a list, have num_sentences elements. each element is [None, sentence_len = sequence_len/num_sentences]
        input_x = tf.stack(input_x, axis=1) # shape:[None, num_sentences, sentence_len = num_sequence_len/num_sentences]
        self.embeded_words = tf.nn.embedding_lookup(self.embedding, input_x) # [None, num_sentences, sentence_len, embed_size]
        embeded_words_reshaped = tf.reshape(self.embeded_words, [-1, self.sentence_len, self.embed_size]) # [batch_size*num_sentences, sentence_len, embed_size]

        # 1.2 forward gru
        hidden_state_forward_list = self.gru_froward_word_level(embeded_words_reshaped)

        # 1.3 backward gru
        # example of reverse anf split
            # a = tf.constant([0,1,2,3,4,5])
            # b = tf.split(a, 3) # a list, 3 elements, each elements is (2,)
            # b.reverse()
            # c = tf.convert_to_tensor(b) #
            # tf.InteractiveSession()
            # print(c.eval()) # [[4 5] [2 3][0 1]]
        embeded_words_reshaped.reverse()
        hidden_state_backward_list = self.gru_froward_word_level(embeded_words_reshaped)
        hidden_state_backward_list.reverse() # reverse twice

        # 1.4 concat forward hidden state and backward hidden state.
        # hidden_state: a list.len:sentence_length,element:[batch_size*num_sentences,hidden_size*2]
        self.hidden_state = [tf.concat([h_forward, h_backward], axis=1) for h_forward, h_backward in
                             zip(hidden_state_forward_list,
                                 hidden_state_backward_list)]  # hidden_state:list,len:sentence_length,element:[batch_size*num_sentences,hidden_size*2]






    def gru_single_step_word_level(self, Xt, h_t_prev):
        """
        single step of gru for word level
        :param Xt: embedding of the current word [batch_size*num_sentences, embed_size]
        :param h_t_prev: the hidden state of the previous state [batch_size*num_sentences, hidden_size]
        :return: h_t: [batch_size*num_sentences,hidden_size]
        """
        # the reset gate
        r_t = tf.nn.sigmoid(tf.matmul(self.Wr, Xt) + tf.matmul(self.Ur, h_t_prev) + self.br)
        # the new memory cell
        h_t_candidate = tf.nn.tanh(tf.matmul(self.Wh, Xt) + tf.multiply(r_t, tf.matmul(self.Uh, h_t_prev)) + self.bh)
        # the update gate
        z_t = tf.nn.sigmoid(tf.matmul(self.Wz, h_t_prev) + tf.matmul(self.Uz, h_t_prev) + self.bz)
        # the new state
        h_t = tf.multiply(1 - z_t, h_t_prev) + tf.multiply(z_t, h_t_candidate)
        return h_t

    def gru_froward_word_level(self, embeded_words):
        """

        :param embeded_words: [batch_size*num_sentences,sentence_length,embed_size]
        :return: a list.length is sentence_length, each element is [batch_size*num_sentences,hidden_size]
        """
        # split embeded_words
        embeded_words_splitted = tf.split(embeded_words, self.sequence_len,
                                          axis=1)  # a list, length is sentence_len, each element is [batch_size*num_sentences, 1, embed_size]
        embeded_words_squeeze = [tf.squeeze(x, axis=1) for x in
                                      embeded_words_splitted]  # a list, length is sentence_len, each element is [batch_size*num_sentences, embed_size]
        # initilization hidden state
        h_t = tf.ones(shape=[self.batch_size * self.num_sentences, self.hidden_size])
        h_t_forward_list = []
        for time_step, Xt in enumerate(embeded_words_squeeze):
            h_t = self.gru_single_step_word_level(Xt, h_t) # update the previous hidden state directly
            h_t_forward_list.append(h_t)
        return h_t_forward_list


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