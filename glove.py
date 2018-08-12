import numpy as np
from data_utils import build_tokens
import tqdm
import os

DEFAULT_PATH = "glove_wv/glove.6B.300d.txt"

def loadWordVec(tokens, filepath=DEFAULT_PATH, dimension=300):
    if os.path.exists("wordvec.npy"):
        wordVec = np.load("wordvec.npy")
    else:
        wordVec = np.zeros((len(tokens), dimension))
        with open(filepath, "r") as ifs:
            print("load wordvec!")
            for line in tqdm.tqdm(ifs):
                line = line.strip()
                if not line:    ## ???
                    continue
                raw = line.split()
                token = raw[0]
                if token not in tokens:  ### 这个词不再词典中，下一个循环
                    continue
                data = [float(x) for x in raw[1:]]
                if len(data) != dimension:
                    raise RuntimeError("wrong number of dimension")
                wordVec[tokens[token]] = np.asarray(data)
    np.save('wordvec.npy', wordVec)
    return wordVec

if __name__ == "__main__":
    tokens = build_tokens()
    wordVec = loadWordVec(tokens)
    print(wordVec.shape)           # (30003, 300)