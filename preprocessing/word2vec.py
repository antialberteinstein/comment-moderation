from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

class Word2VecModel:

    def __init__(self, sentences, vector_size=100, window=5, min_count=1, workers=4):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        
        processed_sentences = [self._process_sentence(sentence) for sentence in sentences]
        
        self._model = Word2Vec(
            sentences=processed_sentences,
            vector_size=self.vector_size, 
            window=self.window,
            min_count=self.min_count,
            workers=self.workers
        )

    def fit(self, new_sentences):
        processed_sentences = [self._process_sentence(sentence) for sentence in new_sentences]
        self._model.build_vocab(processed_sentences, update=True)
        self._model.train(processed_sentences, total_examples=self._model.corpus_count, epochs=self._model.epochs)

    def get_sentence_vector(self, sentence):
        processed_words = self._process_sentence(sentence)
        features = None
        for word in processed_words:
            if word in self._model.wv:
                if features is None:
                    features = self._model.wv[word]
                else:
                    features = features + self._model.wv[word]
        return features / len(processed_words) if features is not None else None

    @staticmethod
    def _process_sentence(sentence):
        if isinstance(sentence, str):
            # If input is a string, tokenize it first
            tokenized_words = word_tokenize(sentence.lower())
        else:
            # If input is already tokenized, just convert to lowercase
            tokenized_words = [word.lower() for word in sentence]
            
        # Filter punctuation and stop words
        filtered_words = [word for word in tokenized_words if word not in string.punctuation 
                        and word not in stop_words]
        
        # Stem words
        stemmer = PorterStemmer()
        stemmed = []
        
        for word in filtered_words:
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
