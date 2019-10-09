from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from torch_rnn_classifier import TorchRNNClassifier
from torch_shallow_neural_classifier import TorchShallowNeuralClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from collections import Counter
import torch.nn as nn
import pandas as pd
import numpy as np
import fastText
import os, io
import vsm
import time


# Helper method to read in hotel data from csv file
def read_data():
    with open(UNBALANCED_REVIEWS_FILE, encoding='UTF-16') as f:
        df = pd.read_csv(f, sep='\t')
    df.rename(columns={'Hotel name': 'hotel', 'user type': 'user', 'room type': 'room'}, inplace=True)
    df.drop(columns=['no'], inplace=True)
    return df

# In this classification problem, we will utilize look at categorizing reviews
# as positive, negative, or neutral. Thus, we use the ternary class function. We also
# define the binary class function below experiment with additionally.
def ternary_class_func(y):
    if y in (1, 2): return 'negative'
    elif y in (4, 5): return 'positive'
    else: return 'neutral'

def binary_class_func(y):
    if y in (1, 2): return 'negative'
    elif y in (4, 5): return 'positive'
    else: return None

# To fit a baseline Logisitc Regression model, we first need to create a feature
# function to evaluate the reviews. We will use a simple unigram feature function.

def unigram_features(row):
    review = row['review']
    review = review.replace(',','').replace('.','').replace('”','').replace('“','').replace('...', '')
    d = dict(Counter(review.split(' ')))
    d['len'] = len(review)
    d['hotel'] = row['hotel']
    d['rating'] = row['rating']
    d['user'] = row['user']
    d['room'] = row['room']
    d['nights'] = row['nights']
    return d

# Finally, we build the dataset, obtaining a vectorized feature matrix
# of the input data for the baseline model.
def build_dataset(df, phi, class_func, vectorizer=None, vectorize=True):
    labels = []
    feat_dicts = []
    raw_examples = []
    for index, row in df.iterrows():
        labels.append(row['rating'])
        feat_dicts.append(phi(row))
        raw_examples.append(row['review'])
    feat_matrix = None
    if vectorize:
        # In training, we want a new vectorizer:
        if vectorizer == None:
            vectorizer = DictVectorizer(sparse=True)
            feat_matrix = vectorizer.fit_transform(feat_dicts)
        # In assessment, we featurize using the existing vectorizer:
        else:
            feat_matrix = vectorizer.transform(feat_dicts)
    else:
        feat_matrix = feat_dicts
    return {'X': feat_matrix,
            'y': labels,
            'vectorizer': vectorizer,
            'raw_examples': raw_examples}

# We will train and predict the baseline using a Logistic Regression Softmax Classifier
def fit_softmax_classifier(X, y):
    mod = LogisticRegression(
        fit_intercept=True,
        solver='liblinear',
        multi_class='ovr')
    mod.fit(X, y)
    return mod

# For simplicity and reproducibility, an experiment method was generated to carry out
# the train/test split and evaluation
def experiment(train, train_func, train_size=0.7):
    X_train = train['X']
    y_train = train['y']
    raw_train = train['raw_examples']
    X_train, X_assess, y_train, y_assess, raw_train, raw_assess = train_test_split(
        X_train, y_train, raw_train,
        train_size=train_size, test_size=None)
    assess = {
        'X': X_assess,
        'y': y_assess,
        'vectorizer': train['vectorizer'],
        'raw_examples': raw_assess}
    mod = train_func(X_train, y_train)
    predictions = mod.predict(X_assess)
    print(classification_report(y_assess, predictions, digits=3))
    return {'model': mod,
            'report': classification_report(y_assess, predictions, digits=3)}

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    count = 0
    for line in fin:
        count += 1
        tokens = line.rstrip().split(' ')
        if tokens[0] in all_words:
            data[tokens[0]] = tokens[1:]
    return data

def fasttext_phi(row):
    review = row['review'].split(' ')
    d = dict()
    for word in review:
        d[word] = data[word]
    
    return d


# Initialize directory location names
DATA_HOME = 'data/'
UNBALANCED_REVIEWS_FILE = os.path.join(DATA_HOME, 'unbalanced-reviews.txt')     # Review scores on a 1-5 scale
FASTTEXT_EMBEDDINGS_BIN = os.path.join(DATA_HOME, 'cc.en.300.bin')
FASTTEXT_EMBEDDINGS_VEC = os.path.join(DATA_HOME, 'cc.en.300.vec')

print("Reading data")
start_time = time.time()
hotel_review_df = read_data()
#hotel_review_df_copy = read_data()
print("_____ {}s Done reading data".format(time.time() - start_time))

hotels = list(set(hotel_review_df['hotel']))
users = list(set(hotel_review_df['user']))
rooms = list(set(hotel_review_df['room']))
nights = list(set(hotel_review_df['nights']))
nights_replace = []
for night in nights:
    for s in night.split():
        if s.isdigit():
            nights_replace.append(int(s))
        if s == '-':
            nights_replace.append(0)
        if s == 'one':
            nights_replace.append(1)
        if s == 'two':
            nights_replace.append(2)


print("_____ Replacing everything with numbers")
start_time = time.time()
hotel_review_df = hotel_review_df.replace(users, list(range(len(users))))
hotel_review_df = hotel_review_df.replace(hotels, list(range(len(hotels))))
hotel_review_df = hotel_review_df.replace(rooms, list(range(len(rooms))))
hotel_review_df = hotel_review_df.replace(nights, nights_replace)
hotel_review_df.head()
print("_____ {}s Done replacing".format(time.time() - start_time))


print("_____ Building baseline_train")
start_time = time.time()
baseline_train = build_dataset(hotel_review_df,
                               unigram_features,
                               ternary_class_func,
                               vectorizer=None,
                               vectorize=True)
print("_____ {}s Done building baseline_train".format(time.time() - start_time))

print("_____ Running the experiment")
start_time = time.time()
baseline_results = experiment(baseline_train, fit_softmax_classifier)
print("_____ {}s Done running the experiment".format(time.time() - start_time))



reviews = hotel_review_df['review']
rev = list(reviews)
all_words = set(' '.join(rev).split(' '))
DATA_HOME = os.path.join('data')
fname = os.path.join(DATA_HOME, 'cc.en.300.vec')

print("_____ Loading fasttext")
start_time = time.time()
data = load_vectors(fname=fname)
print("_____ {}s Done loading fasttext".format(time.time() - start_time))


for word in all_words:
    if word not in data:
        data[word] = np.random.random_sample((300,)) - 0.5

print("_____ Building baseline_train")
start_time = time.time()
baseline_train = build_dataset(hotel_review_df,
                               fasttext_phi,
                               ternary_class_func,
                               vectorizer=None,
                               vectorize=True)
print("_____ {}s Done building baseline_train".format(time.time() - start_time))


print("_____ Running the experiment")
start_time = time.time()
baseline_results = experiment(baseline_train, fit_softmax_classifier)
print("_____ {}s Done running the experiment".format(time.time() - start_time))






# Now we create an RNN classifier. First, we create a new feature function
# that simply returns the list of split words for the model to work with.
def rnn_features(row):
    return row['review'].split(' ')

# We create the dataset, making sure not to vectorize, since we already have our feature
# vectors for the rnn to work with
print("_____ Building baseline_train")
start_time = time.time()
rnn_train = build_dataset(
                          hotel_review_df,
                          rnn_features,
                          ternary_class_func,
                          vectorizer=None,
                          vectorize=False)
print("_____ {}s Done building baseline_train".format(time.time() - start_time))

# We define a function to get the vocabulary for the dataset and
# the rnn model we use
def get_vocab(X, n_words=None):
    wc = Counter([word for example in X for word in example])
    wc = wc.most_common(n_words) if n_words else wc.items()
    vocab = {word for word, count in wc}
    vocab.add("$UNK")
    return sorted(vocab)

# Now create the RNN classifier, adapting the accompanying TorchRNNClassifier
def fit_rnn_classifier(X, y):
    sst_glove_vocab = get_vocab(X, n_words=10)
    #     sst_glove_vocab = get_vocab(X, n_words=10000)
    mod = TorchRNNClassifier(
                             sst_glove_vocab,
                             eta=0.05,
                             embedding=None,
                             batch_size=1000,
                             embed_dim=50,
                             hidden_dim=50,
                             max_iter=5,
                             l2_strength=0.001,
                             bidirectional=True,
                             hidden_activation=nn.ReLU())
    mod.fit(X, y)
    return mod

print("_____ Running the experiment")
start_time = time.time()
rnn_results = experiment(rnn_train, fit_rnn_classifier)
print("_____ {}s Done running the experiment".format(time.time() - start_time))


















