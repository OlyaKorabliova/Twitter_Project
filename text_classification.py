import pickle, nltk, json

with open("test_set.json", 'r') as f:
    test_set = json.load(f)
with open("all_words.json") as g:
    all_words = json.load(g)

with open("my_classifier.pickle", 'rb') as load_pickle:
    classifier = pickle.load(load_pickle)

    print(nltk.classify.accuracy(classifier, test_set))
    classifier.show_most_informative_features(5)

    test_sentence = "Затримано 24 фігуранта, веземо до Києва"
    test_sentence2 = "Україна повертається до європейської родини"
    test_sent_features = {word.lower(): (word in nltk.word_tokenize(test_sentence.lower())) for word in all_words}
    test_sent_features2 = {word.lower(): (word in nltk.word_tokenize(test_sentence2.lower())) for word in all_words}
    print(classifier.classify(test_sent_features))
    print(classifier.classify(test_sent_features2))