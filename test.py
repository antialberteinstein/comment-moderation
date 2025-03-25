from preprocessing.svd_extractor import SVDExtractor
from preprocessing.tcw_builder import TCWBuilder

import numpy as np
import pandas as pd

documents = [
    # Category: Science
    "The Earth orbits the Sun at a speed of approximately 107,000 km per hour.",
    "Albert Einstein developed the theory of relativity in the early 20th century.",
    "Artificial intelligence is rapidly evolving and finds applications in various fields.",

    # Category: Sports
    "Cristiano Ronaldo is one of the greatest football players in the world.",
    "Basketball is a popular team sport in the United States and many other countries."
]

# 0: Science
# 1: Sport
labels = [0, 0, 0, 1, 1]

# builder = TCWBuilder()
# builder.fit_transform(documents, labels)
#
# tcw_matrix = builder.tcw
#
# print('TCW Matrix is:')
# print(tcw_matrix)
#
# extractor = SVDExtractor(k=0.5)
# extractor.fit_transform(tcw_matrix)
#
# features = extractor.features_matrix
#
# print('Features is:')
# print(pd.DataFrame(features))