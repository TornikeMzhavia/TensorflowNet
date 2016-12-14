import nltk
from nltk.tokenize import word_tokenize #words to vector
from nltk.stem import WordNetLemmatizer #clean words from grammar: *ing, *ed...
import numpy as np
import random
import pickle
import os.path
from collections import Counter

lemmitizer = WordNetLemmatizer()
hm_lines = 10000000

def create_lexicon(pos, neg, min_count = 50, max_count = 1000):
    lexicon = Counter([])
    for dir in [pos, neg]:
        with open(dir, 'r') as f:
            for line in f:
                lexicon.update(lemmitizer.lemmatize(w) for w in word_tokenize(line.lower().strip()))

    return [word for word, count in lexicon.items() if(min_count < count < max_count)]

def sample_handling(sample, lexicon, classification):
    featureset = []

    with open(sample, 'r') as f:
        contents = f.readlines()
        for line in contents[:hm_lines]:
            current_words = [lemmitizer.lemmatize(i) for i in word_tokenize(line.lower().strip())]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word in lexicon:
                    features[lexicon.index(word)] += 1

            featureset.append((features, classification))

    print(len(featureset))
    return featureset

def create_feature_sets_and_labels(pos, neg, test_size = 0.1):
    lexicon = create_lexicon(pos, neg)

    features = []
    features.extend(sample_handling(pos, lexicon, [1,0]))
    features.extend(sample_handling(neg, lexicon, [0,1]))

    random.shuffle(features)

    testing_size = int(test_size*len(features))
    features = np.array(features)

    train_x = features[:,0][:-testing_size]
    train_y = features[:,1][:-testing_size]

    test_x = features[:,0][-testing_size:]
    test_y = features[:,1][-testing_size:]

    return train_x, train_y, test_x, test_y


def get_data():
    with open('sentiment_set.pickle', 'rb') as f:
        pickle.load([train_x, train_y, test_x, test_y], f)


def store_data():
    train_x, train_y, test_x, test_y = create_feature_sets_and_labels(os.path.join(os.path.dirname(__file__), 'pos.txt'), 
                                                                      os.path.join(os.path.dirname(__file__), 'neg.txt'))

    with open('sentiment_set.pickle', 'wb') as f:
        pickle.dump([train_x, train_y, test_x, test_y], f)

#store_data()
