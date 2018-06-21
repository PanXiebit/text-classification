# the modules of transformer
# position encoding


import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import xavier_initializer

""" Embddding and softmax
Similarly to other sequence transduction models, we use learned embeddings to convert the input tokens and output
tokens to vectors of dimension d_model. We also use the usual learned linear transformation and softmax function
to convert the decoder output to predicted next-token probabilities. In our model, we share the same weight
matrix between the two embedding layers and the pre-softmax linear transformation, similar to [Using the output
embedding to improve language models](https://arxiv.org/abs/1608.05859). In the embedding layers, we multiply 
those weights by pdmodel.
"""
def embedding_mine(vocab_size,
                   num_units,
                   zero_pad,
                   scale,
                   reuse=None):
    with tf.variable_scope("embedding-layer", reuse=reuse):
        embedding = tf.get_variable("embedding", [vocab_size, num_units],
                                    initializer=xavier_initializer())
        if zero_pad:
            embedding = tf.concat([tf.zeros([1, num_units]),
                                  embedding[1:, :]], axis=0)  # index=0 for nil word
        if scale:
            embedding /= np.sqrt(num_units)
        return embedding


def embedding(inputs,
              vocab_size,
              num_units,
              zero_pad=True,
              scale=True,
              reuse=None):
    """

    :param inputs:  A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`. shape is [batch, sentence_len]
    :param vocab_size: vocabulary size
    :param num_units: Number of embedding hidden units. in the paper, it is called d_model
    :param zero_pad: If True, all the values of the fist row (id 0)
        should be constant zeros.
    :param scale: If True. the outputs is multiplied by sqrt num_units.
    :param reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    :return:
        A `Tensor` with one more rank than inputs's. The last dimensionality
        should be `num_units`.

    """
    with tf.variable_scope("embedding-layer", reuse=reuse):
        embedding = tf.get_variable("embedding", [vocab_size, num_units],
                                    initializer=xavier_initializer())
        if zero_pad:
            embedding = tf.concat([tf.zeros([1, num_units]),
                                  embedding[1:, :]], axis=0)  # index=0 for nil word
        output = tf.nn.embedding_lookup(embedding, inputs)    # [batch, sentence_len, num_units]
        if scale:
            output = output * np.sqrt(num_units)

    return output


"""
we add "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks.
The positional encodings have the same dimension d_model as the embeddings, so that the two can be summed.
There are many choices of positional encodings, learned and fixed [9].
"""

def position_encoding_mine(n_position, d_model):
    """ Init the sinusoid position encoding table.

    :param n_position: the lenght of sentence
    :param d_model: the same with embedding
    :return:
    """
    # keep dim -1 for padding token position encoding zero vector
    # pos=-1 用于 padded zero vector
    encoding = np.zeros([n_position, d_model], np.float32)
    for pos in range(1, n_position):
        for i in range(0, d_model):
            encoding[pos, i] = pos /np.power(10000, 2*i/d_model)

    encoding[1:-2, 0::2] = np.sin(encoding[1:-2, 0::2]) # dim 2i
    encoding[1:-2, 1::2] = np.cos(encoding[1:-2, 1::2]) # dim 2i+1
    return encoding

def positional_encoding(inputs,
                        num_units,
                        zero_pad=True,
                        scale=True,
                        scope="positional_encoding",
                        reuse=None):
    '''Sinusoidal Positional_Encoding.

    Args:
      inputs: A 2d Tensor with shape of (N, T).
      num_units: Output dimensionality
      zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
      scale: Boolean. If True, the output will be multiplied by sqrt num_units(check details from paper)
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
        A 'Tensor' with one more rank than inputs's, with the dimensionality should be 'num_units'
    '''

    N, T = inputs.get_shape().as_list()  # N means batch_size, T means the sentence length.
    with tf.variable_scope(scope, reuse=reuse):
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])  # [N, T]
        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, 2.*i/num_units) for i in range(num_units)]
            for pos in range(T)])                                       # [T, num_units]

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        # Convert to a tensor
        lookup_table = tf.convert_to_tensor(position_enc, dtype=tf.float32)

        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), axis=0)
        outputs = tf.nn.embedding_lookup(lookup_table, position_ind)  # [N, T, num_units]

        if scale:
            outputs = outputs * num_units**0.5

        return outputs, lookup_table

# position_encoding_mine 和 positional_encoding 中其实没啥区别，encoding 都是 [sentence_len, d_model],
# 不过后者直接将 input 作为输入，并与encoding 结合得到 output


def Normalize(inputs,
              epsilon = 1e-8,
              reuse=None):
    """ Applies layer normalization.
    :param inputs:
    :param epsilon:
    :param scope:
    :param reuse:
    :return:
    """
    with tf.variable_scope("bn-normalization", reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, axes=[-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / (variance ** 0.5 + epsilon)
        outputs = tf.to_float(gamma) * normalized + tf.to_float(beta)
    return outputs








def position_wise_feed_forward():
    pass







if __name__ == "__main__":
    tf.InteractiveSession()
    n_pos = 10
    d_model = 10
    input = tf.random_normal([10, 10])
    encoding1 = position_encoding_mine(n_pos, d_model)
    output, encoding2 = positional_encoding(input, d_model)
    print(encoding1 == encoding2.eval())  ## ??
    print("----------")
    print(encoding1)
    print(encoding2.eval())







