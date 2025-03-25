# Import libraries
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
import numpy as np
from collections import Counter
from . import stop_words


class TCWBuilder:
    def __init__(self):
        self._tcw = None
        self._trainset = None
        self._universal_tokens = None

    def fit_transform(self, x):
        self._fit(x)

    @property
    def tcw(self):
        return self._tcw

    def _fit(self, x):
        N = len(x)
        self._trainset = x

        dicts = []
        for tr in x:
            tokenized_words = self._tokenize(tr)
            stemmed_words = self._stem(tokenized_words)
            dct = self._tf(stemmed_words)
            dicts.append(dct)

        token_array = np.array(list(self._union_dicts(dicts)))
        self._universal_tokens = token_array
        M = token_array.size

        df_values = np.array([self._df(dicts, token) for token in token_array])
        global_weights = np.where(df_values == 0, 0, np.log2(N / df_values) + 1)

        # Apply global weights directly to vectors
        tcw_list = [TCWBuilder._convert_document_token_frequencies_to_vector(d, token_array, global_weights) for d in dicts]
        self._tcw = np.array(tcw_list).T

    @staticmethod
    def _convert_document_token_frequencies_to_vector(dtokens, token_list, global_weights=None):
        token_array = np.array(token_list)
        frequencies = np.array([dtokens.get(token, 0) for token in token_array])

        return frequencies * global_weights if global_weights is not None else frequencies

    @staticmethod
    def _tokenize(text):
        tokenized_words = word_tokenize(text.lower())
        return [word for word in tokenized_words if word not in string.punctuation
                and word not in stop_words]

    @staticmethod
    def _stem(tokenized_words):
        stemmer = PorterStemmer()
        stemmed = []

        for word in tokenized_words:
            if not isinstance(word, str) or not word.isalpha():
                continue
            try:
                stemmed.append(stemmer.stem(word))
            except RecursionError:
                print(f'Detect the word {word} cause RecursionError')
                stemmed.append(word)
            except Exception:
                print(f'Unknown error occurred while stemming {word}')
                stemmed.append(word)

        return stemmed

    @staticmethod
    def _convert_to_vector(document, token_list):
        _token_dict = TCWBuilder._tf(
            TCWBuilder._stem(TCWBuilder._tokenize(document)))
        return TCWBuilder._convert_document_token_frequencies_to_vector(_token_dict, token_list)

    @staticmethod
    def _tf(stemmed_words):
        return Counter(stemmed_words)

    @staticmethod
    def _df(stemmed_documents, token):
        return sum(1 for doc in stemmed_documents if token in doc)

    @staticmethod
    def _union_dicts(dicts):
        return set().union(*[d.keys() for d in dicts])
