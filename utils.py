from collections import defaultdict

embedding_file = 'data_zhihu_cup/word_embedding.txt'


def create_vocabulary(embedding_path):
    with open(embedding_path, 'r') as f:
        word2index = {}
        index2word = {}
        count = 0
        lines = f.readlines()
        for line in lines:
            word = line.strip().split()[0]
            word2index[word] = count
            index2word[count] = word
            count += 1
        return word2index, index2word

def load_data(dataset='train'):
    dataset_path = 'data_zhihu_cup/question_{0}_set.txt'.format(dataset)
    with open(dataset_path, 'r') as f:
        for i in range(3):
            line = f.readline().split()
            print(len(line))
            print(line)

load_data()