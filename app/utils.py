import pickle
from fuzzywuzzy import fuzz
import distance
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
import re

w2v = pickle.load(open('model/w2v.pkl', 'rb'))
word2tfidf = pickle.load(open('model/word2tfidf.pkl', 'rb'))

lemmatizer = WordNetLemmatizer()

def preprocess(q):

    # convert to lowercase
    q = q.strip().lower()

    # replace special characters with their names
    q = q.replace("%", " percent")
    q = q.replace("$", " dollar ")
    q = q.replace('₹', " rupee ")
    q = q.replace('€', " euro ")
    q = q.replace("@", " at " )


    # replace [math] with nothing
    q = q.replace('[math]', '')


    # replace large numbers with string equivalents
    q = q.replace(" ,000", "k ")
    q = q.replace(" ,000,000", "m ")
    q = q.replace(" ,000,000,000", "b ")
    q = re.sub(r'([0-9]+)000', r'\1k', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)


    # remove HTML tags
    q = re.sub(r'<[^>]+>', '', q)


    # Decontracting words
    # https://en.wikipedia.org/wiki/Wikipedia%3aList_of_English_contractions
    # https://stackoverflow.com/a/19794953

    contractions = { 
        "ain't": "am not",
        "aren't": "are not",
        "can't": "can not",
        "can't've": "can not have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have"
        }

    q_decontracted = []

    for word in q.split():
        if word in contractions:
            word = contractions[word]
        
        q_decontracted.append(word)


    q = ' '.join(q_decontracted)
    q = q.replace("'ve", " have")
    q = q.replace("n't", " not")
    q = q.replace("'re", " are")
    q = q.replace("'ll", " will")

    # remove punctuation
    q = re.sub(r'[^\w\s]', '', q)

    q = ' '.join([lemmatizer.lemmatize(word) for word in q.split()])

    return q


def sentence_to_tfidf_weighted_word2vec(sentence, word2tfidf, w2v):
    
    weighted_embeddings = []
    for word in sentence:
        if word in w2v.wv.key_to_index and word in word2tfidf:
            embedding = w2v.wv[word] * word2tfidf[word]
            weighted_embeddings.append(embedding)

    if weighted_embeddings:
        return np.mean(weighted_embeddings, axis=0)
    else:
        return np.zeros(w2v.vector_size)
    




def test_get_token_features(q1, q2):

    SAFE_DIV = 0.0001

    token_features = [0.0]*8

    # get tokens in both questions
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features

    STOP_WORDS = stopwords.words('english')

    # get stopwords in both questions
    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])

    # get ordinary words (non-stopwords) in both questions
    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])

    # number of common tokens in both questions
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))

    # number of common stop words in both tokens
    common_stopword_count = len(q1_stops.intersection(q2_stops))

    # number of common words in both tokens
    common_word_count = len(q1_words.intersection(q2_words))

    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count/ (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stopword_count/ (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stopword_count/ (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count/ (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count/ (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[6] = 1 if (q1_tokens[-1] == q2_tokens[-1]) else 0
    token_features[7] = 1 if (q1_tokens[0] == q2_tokens[0]) else 0


    return token_features




def test_get_length_features(q1, q2):

    q1_tokens = q1.split()
    q2_tokens = q2.split()

    length_features = [0.0]*3

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return length_features

    # mean token length
    length_features[0] = (len(q1_tokens) + len(q2_tokens))/2

    # absolute difference in length
    length_features[1] = abs(len(q1_tokens) - len(q2_tokens))

    # longest substring ratio
    substrs = list(distance.lcsubstrings(q1, q2))
    length_features[2] = len(substrs[0]) / ( min(len(q1), len(q2)) + 1)
        
    return length_features




def test_get_fuzzy_features(q1, q2):

    fuzzy_features = [0.0]*4

    fuzzy_features[0] = fuzz.ratio(q1, q2)
    fuzzy_features[1] = fuzz.partial_ratio(q1, q2)
    fuzzy_features[2] = fuzz.token_sort_ratio(q1, q2)
    fuzzy_features[3] = fuzz.token_set_ratio(q1, q2)

    return fuzzy_features


def test_total_words(q1, q2): 
    w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))

    return len(w1) + len(w2)


def test_common_words(q1, q2): 
    w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))

    return len(w1 & w2)



def query_point_creator(q1, q2):
    
    input_query_features = []

    #preprocessing the input questions
    q1 = preprocess(q1)
    q2 = preprocess(q2)

    # token based features
    token_features = test_get_token_features(q1, q2)
    input_query_features.extend(token_features)

    # length based features
    length_features = test_get_length_features(q1, q2)
    input_query_features.extend(length_features)

    # fuzzy features
    fuzzy_features = test_get_fuzzy_features(q1, q2)
    input_query_features.extend(fuzzy_features)

    # basic features
    input_query_features.append(len(q1))
    input_query_features.append(len(q2))
    
    input_query_features.append(len(q1.split()))
    input_query_features.append(len(q2.split()))

    input_query_features.append(test_total_words(q1, q2))
    input_query_features.append(test_common_words(q1, q2))
    input_query_features.append(test_common_words(q1, q2)/test_total_words(q1, q2))

    q1_embedding = sentence_to_tfidf_weighted_word2vec(q1.split(" "), word2tfidf, w2v).reshape(1, -1)
    q2_embedding = sentence_to_tfidf_weighted_word2vec(q2.split(" "), word2tfidf, w2v).reshape(1, -1)

    return np.hstack((np.array(input_query_features).reshape(1, 22), q1_embedding, q2_embedding))
   
  