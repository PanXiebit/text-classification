import os
import re
import numpy as np
from six.moves import range, reduce
from itertools import chain
from collections import defaultdict


def load_task(data_dir, task_id, only_supporting=False):
    """ load the n_th task, there are 20 tasks in total

    :param data_dir:
    :param task_id:
    :param only_supporting:
    :return: a tuple containing the training and testing data for the task.
    """
    assert task_id > 0 and task_id < 21

    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    s = 'qa{}_'.format(task_id)
    train_file = [f for f in files if s in f and 'train' in f][0]
    test_file = [f for f in files if s in f and 'test' in f][0]
    train_data = get_stories(train_file, only_supporting)
    test_data = get_stories(test_file, only_supporting)
    return train_data, test_data

def tokenize(sentence):
    """Return the tokens of a sentence including puunctuation.

    :param sentence:
    :return:
    """
    return [x.strip() for x in re.split('(\w+)?', sentence) if x.strip()]

def get_stories(f, only_supporting=False):
    """
    Given a file name, read the file, retrieve the stories, and then convert the sentences in a single
    stories. If max_lenght is supplied, any stories longer than max_length tokens will be discarded.
    :param f:
    :param only_supporting:
    :return:
    """
    with open(f) as f:
        return parse_stories(f.readlines(), only_supporting=only_supporting)

def parse_stories(lines, only_supporting=False):
    """Parse stories provided in the bAbI tasks format
    If only_supporting is true, only the sentences that support the answer are kept.

    :param lines:
    :param only_supporting:
    :return:
    """
    data = []
    story = []
    for line in lines:
        line = str.lower(line)
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line: # question
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            #a = tokenize(a)
            # answer is one vocabulary even if it's actually multiple words
            a = [a]
            substory = None

            if q[-1] == '?':
                q = q[:-1]

            if only_supporting:
                # only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i-1] for i in supporting]
            else:
                # provide all the substories
                substory = [x for x in story if x] ## if x 去掉 story 中的“”

            # ([['mary', 'moved', 'to', 'the', 'bathroom'], ['john', 'went', 'to', 'the', 'hallway']], ['where', 'is', 'mary'], ['bathroom'])
            data.append((substory, q, a)) # tuple (substory, q, a)
            story.append('')    ### ??? 将query 和 answer 那一行用"" 代替
        else:
            # remove periods
            sent = tokenize(line)
            if sent[-1] == '.':
                sent = sent[:-1]
            story.append(sent)
    return data

def vectorize_data(data, word_idx, sentence_size, memory_size):
    """
    Vectorize stories and queries.

    If a sentence length < sentence_size, the sentence will be padded with 0's.

    If a story lenght < memory_size, the story will be padded with empty memories.

    :param data: a list, it's element is tuple (substory, query, answer)
    :param word_idx: mapping the word to index
    :param sentence_size: the max lenght of sentence
    :param memory_size: the number of sentences in a single story
    :return:
    """
    S = []
    Q = []
    A = []
    for story, query, answer in data: # data's element is a tuple, (substory, query, answer)
        ss = []
        for i, sentence in enumerate(story, 1): # sentence is a list, for example: ['mary', 'moved', 'to', 'the', 'bathroom']
            ls = max(0, sentence_size - len(sentence))
            ## sentence padding
            ss.append([word_idx[w] for w in sentence] + [0] * ls)

        # take only the most recent sentences that fit in memory
        ss = ss[::-1][:memory_size][::-1] ###???

        # make the last word of each sentence the time 'word' which
        # corresponds to vector of lookup table
        for i in range(len(ss)):
            ss[i][-1] = len(word_idx) - memory_size - i

        # padding of memory_size
        lm = max(0, memory_size - len(ss))
        for _ in range(lm):
            ss.append([0] * sentence_size)

        # padding of query
        lq = max(0, sentence_size - len(query))
        q = [word_idx[w] for w in query] + [0] * lq

        y = np.zeros(len(word_idx) + 1) # 0 is reserved for nil word
        for a in answer:
            y[word_idx[a]] = 1

        S.append(ss)  # [None, memeory_size, sentence_size]
        Q.append(q)   # [None, sentence_size]
        A.append(y)   # [None, vocab_size]
    return np.array(S), np.array(Q), np.array(A)


if __name__ == "__main__":
    data_dir = './data/tasks_1-20_v1-2/en'
    task_id = 1
    only_supporting = False
    # 每三句话一个(story, q, a)
    train, test = load_task(data_dir, task_id, only_supporting)









