##


print("start...")
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import optimize_loss
import numpy as np

class Config(object):
    def __init__(self):

        # set hyperparameter
        self.label_size = 20  # 文档分类的总类别数
        self.batch_size = 8
        self.num_sampled = 5
        self.learning_rate = 0.01
        self.vocab_size = 1000
        self.embed_size = 100
        self.sentence_len =5
        self.is_training = True
        self.decay_steps = 1000
        self.decay_rate = 0.9

class fastText:
    def __init__(self):
        self.config = Config()

        # add placeholder
        self.sentence = tf.placeholder(tf.int32, [None, self.config.sentence_len], name="sentence") #(?,5)
        self.labels = tf.placeholder(tf.int32, [None], name="labels")    #(?,) = (batch_size,)

        # forward propagation: computer loss
        self.initilizate_weights()
        self.logits = self.inference()

        if not self.config.is_training:
            return
        self.loss_val = self.loss()

        # backpropagation
        self.global_step = tf.Variable(0, trainable=False, name="Global_step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = self.config.decay_steps, self.config.decay_rate

        # adding train option
        self.train_op = self.train()

        # computer prediction and accuracy
        self.predictions = tf.argmax(self.logits, axis=1, name='predictions') # shape:[None]
        correct_predition = tf.equal(tf.cast(self.predictions, tf.int32), self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(correct_predition, tf.float32), name='Accuracy')


    def initilizate_weights(self):
        """define all weights here"""
        # embedding matrix
        self.Embedding = tf.get_variable("Embedding", [self.config.vocab_size, self.config.embed_size]) # [1000,100]
        self.W = tf.get_variable("W", [self.config.embed_size, self.config.label_size]) # [100, 19]
        self.b = tf.get_variable("b", [self.config.label_size])

    def inference(self):
        """main computer graph here
        1.embedding --> 2.average --> 3.linear classifier
        """
        # 1. get embedding of words in the sentence
        sentence_embeddings = tf.nn.embedding_lookup(self.Embedding, self.sentence) # [None, entence_len, embed_size]=[?,5,100]

        # 2. average vector, to get representation of the sentence
        self.sentence_embeddings = tf.reduce_mean(sentence_embeddings, axis=1, keepdims=True) # [None, 1, embed_size]
        self.sentence_embeddings = tf.squeeze(self.sentence_embeddings, axis=1) # [None, embed_s]

        # 3. linear classifier layer
        logits = tf.matmul(self.sentence_embeddings, self.W) + self.b # [None, labels_size]

        return logits

    def loss(self, l2_lambda=0.01):
        """calculate loss using (NCE) cross entropy here"""
        # computer the average NCE loss for the batch
        # tf.nec_loss randomly draws a new sample of the negative labels each time we caculate the loss
        if self.config.is_training:
            # labels = tf.reshape(self.labels, [-1,]) # [batch_size,1]-->[batch_size,]
            labels = tf.expand_dims(self.labels, axis=1)        # [batch_size,]-->[batch_size,1]
            loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=tf.transpose(self.W),
                               biases=self.b,
                               labels=labels,
                               inputs=self.sentence_embeddings,
                               num_sampled=self.config.num_sampled,
                               num_classes=self.config.label_size,
                               partition_strategy='div')
            )
        else:  # evaluate/inference
            labels_one_hot = tf.one_hot(indices=self.labels, depth=self.config.label_size) # #[batch_size]-->[batch_size,label_size]
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_one_hot, logits=self.logits)
            print("loss0:", loss) # shape=(?, 1999)
            loss = tf.reduce_sum(loss, axis=1)
            print("loss1:", loss)

        loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
        return loss

    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(learning_rate=self.config.learning_rate,
                                                   global_step=self.global_step,
                                                   decay_steps=self.decay_steps,
                                                   decay_rate=self.decay_rate,
                                                   staircase=True)
        train_op = optimize_loss(loss=self.loss_val,
                                 global_step=self.global_step,
                                 learning_rate=learning_rate,
                                 optimizer="Adam")
        # self.train_op = tf.train.AdamOptimizer()
        return train_op

#test started
def test(config=Config()):
    dropout_keep_prob = 1
    fasttext = fastText()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            input_x = np.zeros((config.batch_size, config.sentence_len),dtype=np.int32)
            input_y = np.array([1,0,1,1,1,1,2,3],dtype=np.int32)
            loss, acc, predict,_ = sess.run([fasttext.loss_val, fasttext.accuracy, fasttext.predictions,fasttext.train_op],
                                            feed_dict={fasttext.sentence:input_x, fasttext.labels:input_y})
            print("loss:{0}, acc:{1}".format(loss, acc))

print("ended...")
if __name__ == "__main__":
    my_config = Config()
    from nltk.corpus import reuters
    print(reuters.fileids())
    # test(config=my_config)