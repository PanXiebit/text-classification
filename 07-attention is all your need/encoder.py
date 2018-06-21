# -*- coding: utf-8 -*-

"""
encoder for the transformer:
6 layers.each layers has two sub-layers.
the first is multi-head self-attention mechanism;
the second is position-wise fully connected feed-forward network.
for each sublayer. use LayerNorm(x+Sublayer(x)). all dimension=512.
"""


class Encoder():
    def __init__(self,d_model, d_k, d_v, sequence_len, heads, batch_size,
                 num_layer, Q, K, type='encoder', mask=None, dropout_keep_prob=None,
                 use_reisdual_connection=True):
        """

        :param d_model:
        :param d_k:
        :param d_v:
        :param sequence_len:
        :param heads:
        :param batch_size:
        :param num_layer:
        :param Q:
        :param K:
        :param type:
        :param mask:
        :param dropout_keep_prob:
        :param use_reisdual_connection:
        """
        pass