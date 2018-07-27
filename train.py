import os
import argparse
from data_utils import *
from TextCNN.TextCNN_model import TextCNN
from sklearn.model_selection import train_test_split
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="TextCNN",
                    help="Fasttext | TextRCNN")
args = parser.parse_args()

if not os.path.exists("dbpedia_csv"):
    print("Downloading dbpedia dataset...")
    download_dbpedia()

# build training dataset
print("build training dataset")
WORD_MAX_LEN = 50
word_dict = build_word_dict()
x, y = build_word_dataset("train", word_dict, WORD_MAX_LEN)

train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.1)

# hyperparameters
NUM_CLASS = 14
BATCH_SIZE = 64
NUM_EPOCHS = 10
VOCAB_SIZE = len(word_dict)
LR = 1e-3


with tf.Session() as sess:
    FILTER_SIZE = [5, 6, 7]
    NUM_FILTER = 50
    EMBED_SIZE = 128
    model = TextCNN(filter_sizes=FILTER_SIZE, num_filter=NUM_FILTER, num_classes=NUM_CLASS, vocab_size=VOCAB_SIZE,
                        embed_size=EMBED_SIZE, sequence_len=WORD_MAX_LEN, learning_rate=LR, batch_size=BATCH_SIZE,
                        decay_steps=100, decay_rate=0.95, num_iter=NUM_EPOCHS, multi_label_flag=False,
                        clip_gradients=5.0, dropout_keep_prob=0.5)

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())

    train_batches = batch_iter(train_x, train_y, BATCH_SIZE, NUM_EPOCHS)
    num_batches_per_epoch = (len(train_x)-1)//BATCH_SIZE + 1
    max_accuracy = 0

    for x_batch, y_batch in train_batches:
        train_feed_dict = {
            model.input_x:x_batch,
            model.input_y:y_batch,
            model.is_training:True
        }

        _, step, loss = sess.run([model.train_op, model.global_step, model.loss], feed_dict=train_feed_dict)

        if step % 100 == 0:
            print("step {0}: loss = {1}".format(step, loss))

        if step % 2000 == 0:
            valid_batches = batch_iter(valid_x, valid_y, BATCH_SIZE, 1)
            sum_accuracy, cnt = 0, 0
            # Test accuracy with validation data for each epoch
            for valid_x_batch, valid_y_batch in valid_batches:
                valid_feed_dict = {
                    model.input_x:valid_x_batch,
                    model.input_y:valid_y_batch,
                    model.is_training:False
                }

                accuracy = sess.run(model.accuracy, feed_dict=valid_feed_dict)
                sum_accuracy += accuracy
                cnt += 1

            valid_accuracy = sum_accuracy / cnt
            print("\nValidation Accuracy = {1}\n".format(step // num_batches_per_epoch, sum_accuracy / cnt))

            # Save model
            if valid_accuracy > max_accuracy:
                max_accuracy = valid_accuracy
                saver.save(sess, "{0}/{1}.ckpt".format(args.model, args.model), global_step=step)
                print("Model is saved.\n")








