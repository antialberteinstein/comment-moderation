# Import libraries
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
import string
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from nltk.corpus import stopwords

# Convert stop_words to a set for faster lookup.
stop_words = set(stopwords.words('english'))


class TrainingSet:
    def __init__(self, text, category):
        self._text = text
        self._category = category

    @property
    def text(self):
        return self._text

    @property
    def category(self):
        return self._category


class TextMachine:
    def __init__(self):
        self._tcw = None
        self._trainset = None
        self._universal_tokens = None
        self._probabilities = None

    def _fit(self, x, y):
        N = len(y)
        training_sets = [TrainingSet(x[i], y[i]) for i in range(len(x))]
        self._trainset = training_sets

        dicts = []
        for tr in training_sets:
            tokenized_words = self._tokenize(tr.text)
            stemmed_words = self._stem(tokenized_words)
            dct = self._tf(stemmed_words)
            dicts.append(dct)

        token_array = np.array(list(self._union_dicts(dicts)))
        self._universal_tokens = token_array
        M = token_array.size

        df_values = np.array([self._df(dicts, token) for token in token_array])
        global_weights = np.where(df_values == 0, 0, np.log2(N / df_values) + 1)

        # Apply global weights directly to vectors
        tcw_list = [self._convert_document_token_frequencies_to_vector(d, token_array, global_weights) for d in dicts]
        self._tcw = np.array(tcw_list).T

    def _first_measure(self, q, a):
        return np.dot(q, a) / (np.linalg.norm(q) * np.linalg.norm(a))

    def _second_measure(self, q, a):
        M = q.size

        x = 0
        y = 0
        z = 0
        t = 0
        # y
        yy = 0
        # z

        for i in range(M):
            x += q[i] * a[i]
            y += q[i]
            z += a[i]
            t += q[i] ** 2
            yy += a[i] ** 2

        result = (M * x - y * z)/np.sqrt((M * t - y ** 2)(M * yy - z ** 2))

        return result


    def _predict(self, x):
        vcts = [self._convert_to_vector(input_text, self._universal_tokens) for input_text in x]

        self._probabilities = []
        for vct in vcts:
            _prob = defaultdict(float)
            counts = defaultdict(int)

            for i, train in enumerate(self._trainset):
                _prob[train.category] += self._first_measure(vct, self._tcw.T[i])
                counts[train.category] += 1

            for key in _prob:
                _prob[key] /= counts[key] if counts[key] != 0 else 1  # Avoid division by zero

            self._probabilities.append(_prob)

        return [max(prob, key=prob.get) for prob in self._probabilities]

    @property
    def probabilities(self):
        return pd.DataFrame(self._probabilities)

    def fit(self, x, y):
        self._fit(x, y)

    def predict(self, x):
        return self._predict(x)

    def _convert_document_token_frequencies_to_vector(self, dtokens, token_list, global_weights=None):
        token_array = np.array(token_list)
        frequencies = np.array([dtokens.get(token, 0) for token in token_array])

        return frequencies * global_weights if global_weights is not None else frequencies

    def _convert_to_vector(self, document, token_list):
        _token_dict = self._tf(self._stem(self._tokenize(document)))
        return self._convert_document_token_frequencies_to_vector(_token_dict, token_list)

    def _tokenize(self, text):
        tokenized_words = word_tokenize(text.lower())
        return [word for word in tokenized_words if word not in string.punctuation]

    def _stem(self, tokenized_words):
        stemmer = PorterStemmer()
        return [stemmer.stem(word) for word in tokenized_words]

    def _tf(self, stemmed_words):
        return Counter(stemmed_words)

    def _df(self, stemmed_documents, token):
        return sum(1 for doc in stemmed_documents if token in doc)

    def _union_dicts(self, dicts):
        return set().union(*[d.keys() for d in dicts])
