# import nltk
import random
from nltk.probability import FreqDist
# from nltk.classify import NaiveBayesClassifier
from ukrainian_stemmer import *
from nltk.tokenize import TweetTokenizer, RegexpTokenizer, word_tokenize
import re
import pickle, json
from stop_words import get_stop_words
from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob


all_stopwords = []
all_stopwords.extend(get_stop_words("ukrainian"))
all_stopwords.extend(get_stop_words("english"))
all_stopwords.extend(get_stop_words("russian"))


################################## READING JSONs #######################################
def preprocess(sentence):
    global all_stopwords
    sentence = sentence.lower()
    tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
    tokens = tokenizer.tokenize(sentence)
    # filtered_words = [w for w in tokens if not w in all_stopwords]
    return " ".join(tokens) #filtered_words)


def read_json(filename):
    global all_stopwords
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.read()
        lines_lst = lines.split('"]["')
    changes = []
    tokenizer = RegexpTokenizer(r'\w+')
    for i in lines_lst:
        i = " ".join(i.split("\\n"))
        x = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', i)
        for r in x:
            i = i.replace(r, "")
        i = ''.join([k for k in i if not k.isdigit()])
        i = preprocess(i)
        tokens = tokenizer.tokenize(i)
        filtered_words = [w for w in tokens if w not in all_stopwords]
        changes.append(" ".join(filtered_words))
        if "" in changes: changes.remove('')
    return changes
########################################################################################


################## CREATE DICTIONARY OF DATA FROM JSONs ###############################
def create_dict(filename="list_of_JSONs.txt"):
    categories = dict()
    with open(filename, 'r') as f:
        for i in f.readlines():
            if " = " in i:
                tup = i.strip().split(" = ")
                categories[tup[0]] = read_json("JSONs/" + tup[1])
    return categories

categories = create_dict("t.txt")
########################################################################################


#################################### STEMMING ##########################################
print("1")
stemmer = UkrainianStemmer()

def stemming():
    class_words = {}
    cl_words = {}

    classes = list(set([a for a in categories.keys()]))
    for c in classes:
        class_words[c] = []
        cl_words[c] = []
    for data in categories.keys():
        for i in categories[data]:
            l = ''
            for word in word_tokenize(i):
                stemmed_word = stemmer.stem_word(word)
                cl_words[data].extend(stemmed_word)
                l += stemmed_word + " "
            l = l.strip()
            class_words[data].append([l])

    common = []
    for i in cl_words.keys():
        common.extend(cl_words[i])
    return class_words, common
########################################################################################

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['{}'.format(word[0])] = (word[0] in document_words)
    return features

common_dict, common_lst = stemming()
print("2")
documents = [(k, i) for i in common_dict.keys() for j in common_dict[i] for k in j]

random.shuffle(documents)
print("3")

all_words = FreqDist(w for w in common_lst)
word_features = all_words.most_common(5000)

n = len(documents)
train, test = documents[:n // 2], documents[n // 2:]
print("4")
cl = NaiveBayesClassifier(train)
print("5")
with open("my_classifier.pickle", "wb") as dump_pickle:
    pickle.dump(cl, dump_pickle)

with open("test_set.json", 'w') as file_test:
    json.dump(test, file_test)
with open("all_words.json", 'w') as file_all_words:
    json.dump(all_words, file_all_words)