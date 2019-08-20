import string
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

class prep:
    def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
        # Clean the text, with the option to remove stopwords and to stem words.
        # Convert words to lower case and split them
        text = text.lower().split()
        # Optionally, remove stop words
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            text = [w for w in text if not w in stops]
        text = " ".join(text)
        # Clean the text
        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"\-", " - ", text)
        text = re.sub(r"\=", " = ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" u s ", " american ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e - mail", "email", text)
        text = re.sub(r"j k", "jk", text)
        text = re.sub(r"\s{2,}", " ", text)
        # Optionally, shorten words to their stems
        if stem_words:
            text = text.split()
            stemmer = SnowballStemmer('english')
            stemmed_words = [stemmer.stem(word) for word in text]
            text = " ".join(stemmed_words)
        # Return a list of words
        return text

    def sentiment_to_label(text, sentiment_corpus):
        label = 0
        for i in range(len(sentiment_corpus)):
            if text==sentiment_corpus[i]:
                label = i
        return label

    def word2vec_GloVe(GloVe_file, word2vec_file, word_index, MAX_NB_WORDS=200000, EMBEDDING_DIM=300):
        # load GloVe and make Embedding Matrix
        glove2word2vec(GloVe_file, word2vec_file)
        word2vec = KeyedVectors.load_word2vec_format(word2vec_file)

        nb_words = min(MAX_NB_WORDS, len(word_index))+1
        embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM), dtype='float32')
        for word, i in word_index.items():
            if word in word2vec.vocab:
                embedding_matrix[i] = word2vec.word_vec(word)
        return embedding_matrix
