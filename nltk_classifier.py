import nltk
import random
from ukrainian_stemmer import *
from nltk.tokenize import TweetTokenizer
import re
from stop_words import get_stop_words


stemmer = UkrainianStemmer()

all_stopwords = []
all_stopwords.extend(get_stop_words("ukrainian"))
all_stopwords.extend(get_stop_words("english"))
all_stopwords.extend(get_stop_words("russian"))

def preprocess(sentence):
    global all_stopwords
    sentence = sentence.lower()
    tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
    tokens = tokenizer.tokenize(sentence)
    filtered_words = [w for w in tokens if not w in all_stopwords]
    return " ".join(filtered_words)

def read_json(filename):
    global all_stopwords
    f = open(filename, 'r', encoding="utf-8")
    lines = f.read()
    lines_lst = lines.split('"]["')
    changes = []
    tokenizer = nltk.RegexpTokenizer(r'\w+')
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

schur = read_json("dataMichaelSchur.json")
poroshenko = read_json("dataPetroPoroshenko.json")

categories = {"poroshenko": poroshenko, "schur": schur}

########################## STEMMING ####################################################
class_words = {}

classes = list(set([a for a in categories.keys()]))
for c in classes:
    class_words[c] = []

for data in categories.keys():
    for i in categories[data]:
        for word in nltk.word_tokenize(i):
            stemmed_word = stemmer.stem_word(word)
            class_words[data].extend([stemmed_word])

common = []
for i in class_words.keys():
    common.extend(class_words[i])
########################################################################################

documents = [(j.split(), i) for i in categories.keys() for j in categories[i]]
for i in categories.keys():
    for j in categories[i]:
        t = (j.split(), i)
        documents.append(t)
random.shuffle(documents)
all_words = nltk.FreqDist(w for w in common)
word_features = all_words.most_common(2000)

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word[0])] = (word[0] in document_words)
    return features

featuresets = [(document_features(d), c) for (d,c) in documents]
n = len(featuresets)
train_set, test_set = featuresets[n // 2:], featuresets[:n // 2]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))
classifier.show_most_informative_features(5)

# test_sentence = "Слава Україні!"
test_sentence2 = "Україна повертається до європейської родини"
# test_sent_features = {word.lower(): (word in nltk.word_tokenize(test_sentence.lower())) for word in all_words}
test_sent_features2 = {word.lower(): (word in nltk.word_tokenize(test_sentence2.lower())) for word in all_words}
# print(classifier.classify(test_sent_features))
print(classifier.classify(test_sent_features2))
