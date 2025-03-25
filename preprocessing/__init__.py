# Downloading some neccessary library of nltk
import nltk
from nltk.corpus import stopwords

def download_library_if_not_found(library_path_to_check, library_name_to_download):
    try:
        nltk.data.find(library_path_to_check)
    except LookupError:
        print(f'Missing {library_path_to_check}, starting to download {library_name_to_download}')
        nltk.download(library_name_to_download)

# punkt
download_library_if_not_found('tokenizers/punkt', 'punkt')

# stopwords
download_library_if_not_found('corpora/stopwords', 'stopwords')


# Create a set of stop words
stop_words = set(stopwords.words('english'))