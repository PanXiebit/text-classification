from tensorflow.contrib.layers import xavier_initializer


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