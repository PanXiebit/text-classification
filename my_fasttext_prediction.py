# -*- coding: utf-8 -*-
# prediction using model.
# process --> 1. load data (X:list of int, y:int) 2. create Session 3. feed data 4. predict

from importlib import reload

import sys
reload(sys)
# sys.setdefaultencoding('utf8')
# Python3字符串默认编码unicode, 所以sys.setdefaultencoding也不存在了

import tensorflow as tf
import numpy as np
from my_fasttext import fastText
from data_util_zhihu import load_data_predict,load_final_test_data,create_voabulary,create_voabulary_label
from tflearn.data_utils import to_categorical, pad_sequences
import os
import codecs

# configuration
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("label_size", 1999, "number of label")
tf.app.flags.DEFINE_float('learning_rate', 0.01, "learning rate")
tf.app.flags.DEFINE_integer('batch_size', 512, "Batch size for training")
tf.app.flags.DEFINE_integer("decay_steps", 5000, "how many steps before decay learning rate")
tf.app.flags.DEFINE_float('decay_rate', 0.9, "rate of decay for learning rate.")
tf.app.flags.DEFINE_integer('num_sampled', 10, "number of noise sampling")
tf.app.flags.DEFINE_string('ckpt_dir','fast_text_checkpoint/','checkpoint location for the model')
tf.app.flags.DEFINE_integer('sentence_len',300,"max sentence length")
tf.app.flags.DEFINE_integer('embed_size',100,"embedding size")
tf.app.flags.DEFINE_boolean('is_training', False, "is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer('num_epoch', 15, "number of epoches")
tf.app.flags.DEFINE_integer("validate_every", 3, "Validate every validate very epoch")
tf.app.flags.DEFINE_string("predict_target_file", "fast_text_checkpoint/zhihu_result_fasttext.csv","target file path for final prediction")
tf.app.flags.DEFINE_string("predict_source_file",'test-zhihu-forpredict-v4only-title.txt',"target file path for final prediction")

# load data
def main():
    # load data with vocabulary of words and labels
    vocabulay_word2index, vocabulary_index2word = create_vocabulary()
    vocab_size = len(vocabulary_index2word)

