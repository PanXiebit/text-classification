import tensorflow as tf
from data_utils import build_word_dataset, build_tokens
from data_utils import batch_iter
import tqdm


flags = tf.app.flags
flags.DEFINE_integer("sentence_len", 30, "the max length a sentence in dataset")
FLAGS = flags.FLAGS

tokens = build_tokens()  # len(tokens)=30003
X_test, y_test = build_word_dataset("test", tokens, FLAGS.sentence_len)   # [70000, 50]
tf.reset_default_graph()
with tf.Session() as sess:
    saver = tf.train.import_meta_graph("./temp/model/save-1000.meta")
    saver.restore(sess, tf.train.latest_checkpoint("./temp/model"))
    graph = tf.get_default_graph()
    accuracy = graph.get_tensor_by_name("accuracy:0")
    X = graph.get_tensor_by_name("input_sentence:0")
    y = graph.get_tensor_by_name("input_label:0")
    is_training = graph.get_tensor_by_name("is_training:0")

    test_accuracy = 0
    test_data = batch_iter(X_test, y_test, 10, 1)
    for test_batch_X, test_batch_y in tqdm.tqdm(test_data):
        acc = sess.run(accuracy, feed_dict={X:test_batch_X, y:test_batch_y, is_training:False})
        test_accuracy += acc
    test_accuracy /= (len(X_test)/10)
    print("Test accuracy is {}".format(test_accuracy))









