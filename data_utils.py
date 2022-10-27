import os
import wget
import tqdm
import tarfile
import collections
import pandas as pd
import pickle
import numpy as np
from preprocessing import TextPreProcessing
from sklearn.cross_validation import train_test_split

Train_path = "dataset/train.csv"
Test_path = "dataset/test.csv"

def download_dbpedia():
    dbpedia_url = "https://github.com/le-scientifique/torchDatasets/raw/master/dbpedia_csv.tar.gz"

    wget.download(dbpedia_url)
    with tarfile.open("dbpedia_csv.tar.gz", "r:gz") as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar)

# 建立词典
def build_tokens():
    if not os.path.exists("word_dict.pickle"):
        train_df = pd.read_csv(Train_path, names=["class", "title", "content"])
        contents = train_df["content"]

        words = []
        print("build vocabuary!")
        for content in tqdm.tqdm(contents):
            for word in TextPreProcessing.lemma(content):
                words.append(word)

        word_counter = collections.Counter(words).most_common(n=30000) # 选出最常见的30k词
        word_dict = dict()
        word_dict["<pad>"] = 0
        word_dict["<unk>"] = 1
        word_dict["<eos>"] = 2
        for word, _ in word_counter:
            word_dict[word] = len(word_dict)  # 逐一添加单词进入词典

        with open("word_dict.pickle", "wb") as f:
            pickle.dump(word_dict, f)

    else:
        with open("word_dict.pickle", "rb") as f:
            word_dict = pickle.load(f)

    return word_dict

# 建立数据集
def build_word_dataset(step, word_dict, document_max_len):
    if not os.path.exists("./dataset/{}_X_{}.npy".format(step, document_max_len)):
        if step == "train":
            df = pd.read_csv(Train_path, names=["class", "title", "content"])
        else:
            df = pd.read_csv(Test_path, names=["class", "title", "content"])

        print("build {} dataset!".format(step))

        # Shuffle dataframe, frac表示axis
        # df = df.sample(frac=1)
        # 每一行进行特殊符号处理，以及分词
        x = list(map(lambda d: TextPreProcessing.lemma(d), df["content"]))
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

        # save
        X = np.array(x)
        y = np.array(y)
        np.save("./dataset/{}_X_{}.npy".format(step, document_max_len), X)
        np.save("./dataset/{}_y_{}.npy".format(step, document_max_len), y)

        print("build {} dataset finished!".format(step))

    else:
        X = np.load("./dataset/{}_X_{}.npy".format(step, document_max_len))
        y = np.load("./dataset/{}_y_{}.npy".format(step, document_max_len))

    return X, y

# 迭代输入数据
def batch_iter(X, y, batch_size, num_epochs):
    num_batches_per_epoch = (len(X) - 1) // batch_size + 1
    for epoch in range(num_epochs):
        index = np.arange(len(X))
        np.random.shuffle(index)
        X = X[index]
        y = y[index]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(X))
            yield X[start_index:end_index], y[start_index:end_index]

if __name__ == "__main__":
    pass
