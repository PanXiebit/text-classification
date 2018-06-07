import tensorflow as tf

def word_attention_level(config, hidden_state_list):
    """
    :param hidden_state_list: a list, sentence_len elements, and each element is [batch_size*num_sentences, hidden_size*2]
    :return:
    """
    # tf.stack()
    # Given a list of length `N` of tensors of shape `(A, B, C)`;
    # if `axis == 0` then the `output` tensor will have the shape `(N, A, B, C)`.
    # if `axis == 1` then the `output` tensor will have the shape `(A, N, B, C)`.
    hidden_state_1 = tf.stack(hidden_state_list, axis=1)  # [batch_size*num_sentences, sentence_len, hidden_size*2]

    # 为了方便所有的hidden_state与矩阵config.Ww相乘，将其shape转换为 [batch_size*num_sentences*sentence_len, hidden_size*2]
    hidden_state_2 = tf.reshape(hidden_state_1,
                                [-1, config.hidden_size * 2])  # [batch_size*num_sentences*sentence_len, hidden_size*2]

    # 1) one-layer MLP: hidden_representation is the u_{it}
    u_it = tf.nn.tanh(tf.matmul(hidden_state_2, config.Ww) + config.bw,
                      name='hidden_representation')  # [batch_size*num_sentences*sentence_len, hidden_size*2]
    u_it = tf.reshape(u_it, [-1, config.sentence_len,
                             config.hidden_size * 2])  # [batch_size*num_sentences, sentence_len, hidden_size*2]

    # 2) get the importance of the word as the similarity of u_{it} with a word level context vector u_w

    # config.uw是共享的，并且是需要训练的参数。与所有的 hidden state 矩阵相乘，求得相似度
    # u_it.shape=[batch_size*num_sentences, sentence_len, hidden_size*2]  config.uw.shape=[config.hidden_size*2]
    similarity_with_uit_uw = tf.reduce_sum(tf.multiply(u_it, config.uw), axis=2,
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


def sentence_attention_level(config, hidden_state_sentence_list):
    """
    :param hidden_state_sentence_list: a list, length is num_sentence, each element is [bath_size, hidden_size*2]
    :return:
    """
    hidden_state_sentence_1 = tf.stack(hidden_state_sentence_list, axis=1)  # [bath_size, num_sentences, hidden_size*2]
    # attention 是可以批量处理的，
    hidden_state_sentence_2 = tf.reshape(hidden_state_sentence_1,
                                         [-1, config.hidden_size * 2])  # [bath_size*num_sentences, hidden_size*2]

    # one MLP layer, hidden_sentence_representation is the u_{i}
    u_i = tf.nn.tanh(tf.matmul(hidden_state_sentence_2, config.Ws) + config.bs,
                     name='hidden_sentence_representation')  # [bath_size*num_sentences, hidden_size*2]
    u_i = tf.reshape(u_i, [config.batch_size, config.num_sentences,
                           config.hidden_size * 2])  # [bath_size, num_sentences, hidden_size*2]

    # the similarity
    similarity_ui_uw = tf.reduce_sum(tf.multiply(u_i, config.us), axis=2, keepdims=False)  # [batch_size, num_sentences]
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