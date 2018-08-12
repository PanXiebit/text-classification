import nltk
import re
from nltk.stem import SnowballStemmer  # 词干提取
from nltk.stem import WordNetLemmatizer # 词形还原

class TextPreProcessing(object):
    def __init__(self):
        pass

    @staticmethod
    def clean_text(text):
        text = re.sub(r"can\'t", "can not", text)
        text = re.sub(r"cannot", "can not ", text)
        text = re.sub(r"what\'s", "what is", text)
        text = re.sub(r"What\'s", "what is", text)
        text = re.sub(r"\'ve ", " have ", text)
        text = re.sub(r"n\'t", " not ", text)
        text = re.sub(r"i\'m", "i am ", text)
        text = re.sub(r"I\'m", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r"c\+\+", "cplusplus", text)
        text = re.sub(r"c \+\+", "cplusplus", text)
        text = re.sub(r"c \+ \+", "cplusplus", text)
        text = re.sub(r"c#", "csharp", text)
        text = re.sub(r"f#", "fsharp", text)
        text = re.sub(r"g#", "gsharp", text)
        text = re.sub(r" e mail ", " email ", text)
        text = re.sub(r" e \- mail ", " email ", text)
        text = re.sub(r" e\-mail ", " email ", text)
        text = re.sub(r",000", '000', text)
        text = re.sub(r"'s", " is", text)                     # example he's -> he is

        # spelling correction
        text = re.sub(r"1", " one ", text)
        text = re.sub(r"2", " two ", text)
        text = re.sub(r"3", " three ", text)
        text = re.sub(r"4", " four ", text)
        text = re.sub(r"5", " five ", text)
        text = re.sub(r"6", " six ", text)
        text = re.sub(r"7", " seven ", text)
        text = re.sub(r"8", " eight ", text)
        text = re.sub(r"9", " nine ", text)

        # symbol replacement
        text = re.sub(r"&", " and ", text)
        text = re.sub(r"\|", " or ", text)
        text = re.sub(r"=", " equal ", text)
        text = re.sub(r"\+", " plus ", text)
        text = re.sub(r"₹", " rs ", text)      # 测试！
        text = re.sub(r"\$", " dollar ", text)

        text = ' '.join(text.split())   # list to string
        return text

    @staticmethod
    def stem(sent):
        stemmer = SnowballStemmer("english")
        sent = [stemmer.stem(word) for word in nltk.word_tokenize(TextPreProcessing.clean_text(sent).lower())]
        return sent
    @staticmethod
    def lemma(sent):
        wnl = WordNetLemmatizer()
        sent = [wnl.lemmatize(word) for word in nltk.word_tokenize(TextPreProcessing.clean_text(sent).lower())]
        return sent

if __name__=="__main__":
    text = "He's my uncle, and he's 29 year old. He loves me very much!"
    prepro = TextPreProcessing
    text = prepro.lemma(text)    # list
    print(text)