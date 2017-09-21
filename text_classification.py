import pickle, nltk, json

with open("test_set.json", 'r') as f:
    test_set = json.load(f)
with open("all_words.json") as g:
    all_words = json.load(g)

with open("my_classifier.pickle", 'rb') as load_pickle:
    classifier = pickle.load(load_pickle)
    
    print(classifier.classify("Затримано 24 фігуранта, веземо до Києва"))
    print(classifier.classify("Україна повертається до європейської родини"))
