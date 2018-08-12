from model.TextCNN import TextCNN
from data_utils import batch_iter, build_tokens, build_word_dataset
import tensorflow as tf
from sklearn.cross_validation import train_test_split



flags = tf.app.flags
flags.DEFINE_integer("sentence_len", 30, "the max length a sentence in dataset")
flags.DEFINE_integer("embed_size", 300, "the dimension of word vector")
flags.DEFINE_integer("num_classes", 14, "the number of classes")
flags.DEFINE_float("l2_lambda", 0.5, "the coefficient of l2 regularization")
flags.DEFINE_float("learning_rate", 0.01, "the learning rate")
flags.DEFINE_list("kernels_size", [3,4,5], "the sizes of kernels")
flags.DEFINE_integer("filter_nums", 100, "the number of feature maps")
flags.DEFINE_boolean("static", False, "whether the embedding is static or non-static")
flags.DEFINE_float("keep_prob", 0.5, "the probability of dropout to keep")
flags.DEFINE_integer("batch_size", 60, "the size of one batch")
flags.DEFINE_integer("batch_size_val", 1000, "the size of one validation batch data")
flags.DEFINE_integer("num_epochs", 10, "the number of epochs")
FLAGS = flags.FLAGS


# vocabulary
tokens = build_tokens()  # len(tokens)=30003

# 模型图
# tf.reset_default_graph()
model = TextCNN(tokens, FLAGS.sentence_len, FLAGS.embed_size, FLAGS.num_classes,
                FLAGS.l2_lambda, FLAGS.learning_rate, FLAGS.kernels_size,
                FLAGS.filter_nums, FLAGS.static, FLAGS.keep_prob)

# 词典, 整理数据集
X, y = build_word_dataset("train", tokens, FLAGS.sentence_len)            # [560000, 50]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)


saver = tf.train.Saver()
merged = tf.summary.merge_all()
with tf.Session() as sess:

    train_writer = tf.summary.FileWriter("./temp/v2/graph/train", graph=sess.graph)
    val_writer = tf.summary.FileWriter("./temp/v2/graph/val")

    sess.run(tf.global_variables_initializer())

    train_data = batch_iter(X_train, y_train, FLAGS.batch_size, FLAGS.num_epochs) # (448000/60)=7466
    val_data = batch_iter(X_val, y_val, FLAGS.batch_size_val, FLAGS.num_epochs) # (112000/10000)*200=11.2*200=2240 # 迭代2240步就能计算一个val准确率
    best_val_acc = 0
    count = 0
    for i, (train_batch_X, train_batch_y) in enumerate(train_data):
        summary, _,step, loss, acc = sess.run([merged, model.train_op, model.global_step, model.loss, model.accuracy],
                                      feed_dict={model.X:train_batch_X,
                                                 model.y:train_batch_y,
                                                 model.is_training:True})
        if step % 100 == 0:
            print("step {}, loss is {}, acc is {}".format(step, loss, acc))
            train_writer.add_summary(summary, step)

        if step % 200 == 0:
            summary_val, val_loss, val_acc = sess.run([merged, model.loss, model.accuracy],
                                         feed_dict={model.X:X_val,
                                                    model.y:y_val,
                                                    model.is_training:False})
            print("step {}, val_loss is {},val_acc is {}\n".format(step, val_loss, val_acc))
            val_writer.add_summary(summary_val, step)
            saver.save(sess, "./temp/v2/model/", global_step=1000)

            # early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            else:
                count += 1
            if count > 5:
                break

train_writer.close()
val_writer.close()
