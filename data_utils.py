import os
import wget
import tarfile
import re
from nltk.tokenize import word_tokenize
import collections
import pandas as pd
import pickle
import numpy as np

Train_path = "dbpedia_csv/train.csv"
Test_path = "dbpedia_csv/test.csv"

def download_dbpedia():
    dbpedia_url = "https://github.com/le-scientifique/torchDatasets/raw/master/dbpedia_csv.tar.gz"

    wget.download(dbpedia_url)
    with tarfile.open("dbpedia_csv.tar.gz", "r:gz") as tar:
        tar.extractall()

# 简单的预处理，还可以去掉停用词等等。可以比较下最后的效果
def clean_str(text):
    # 正则化处理特殊字符
    text = re.sub(r"[^A-Za-z0-9(),!?\'`\"]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.strip().lower()

    return text

def build_word_dict():
    if not os.path.exists("word_dict.pickle"):
        train_df = pd.read_csv(Train_path, names=["class", "title", "content"])
        contents = train_df["content"]

        words = []
        for content in contents:
            for word in word_tokenize(clean_str(content)):
                words.append(word)

        word_counter = collections.Counter(words).most_common()
        word_dict = dict()
        word_dict["<pad>"] = 0
        word_dict["<unk>"] = 1
        word_dict["<eos>"] = 2
        for word, _ in word_counter:
            word_dict[word] = len(word_dict)

        with open("word_dict.pickle", "wb") as f:
            pickle.dump(word_dict, f)

    else:
        with open("word_dict.pickle", "rb") as f:
            word_dict = pickle.load(f)

    return word_dict

def build_word_dataset(step, word_dict, document_max_len):
    if step == "train":
        df = pd.read_csv(Train_path, names=["class", "title", "content"])
    else:
        df = pd.read_csv(Test_path, names=["class", "title", "content"])

    # Shuffle dataframe, frac表示axis
    df = df.sample(frac=1)
    # 每一行进行特殊符号处理，以及分词
    x = list(map(lambda d: word_tokenize(clean_str(d)), df["content"]))
    # 两个函数的嵌套，对每一行找到对应在word_dict中的index，不存在的默认为 word_dict['<unk>']
    x = list(map(lambda d: list(map(lambda w: word_dict.get(w, word_dict["<unk>"]), d)), x))
    # 每一行末尾加上 <eos>
    x = list(map(lambda d: d + [word_dict["<eos>"]], x))
    # 序列长度最长 document_max_len
    x = list(map(lambda d: d[:document_max_len], x))
    # 不足这个长度的补 <pad>
    x = list(map(lambda d: d + (document_max_len - len(d)) * [word_dict["<pad>"]], x))
    
    
    # label
    y = list(map(lambda d: d - 1, list(df["class"])))

    return x, y

# 迭代输入数据
def batch_iter(inputs, outputs, batch_size, num_epochs):
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(inputs))
            yield inputs[start_index:end_index], outputs[start_index:end_index]
