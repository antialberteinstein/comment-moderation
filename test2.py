from preprocessing.word2vec import Word2VecModel

sentences = [
    ["this", "is", "the", "first", "sentence"],
    ["this", "is", "the", "second", "sentence"],
    ["this", "is", "the", "third", "sentence"],
    ["this", "is", "the", "fourth", "sentence"],
    ["this", "is", "the", "fifth", "sentence"],
]

model = Word2VecModel(sentences)

