#%clear

import pandas as pd
import numpy as np
import concurrent.futures as conc

import gensim
import nltk
import timeit
import random
#nltk.download('all')

from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim import corpora, models
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *


#import Data
def import_data():
    data = pd.read_csv("abcnews-date-text.csv", error_bad_lines=False)
    data_text = data[['headline_text']]
    data_text['index'] = data_text.index
    
    documents = data_text
    return documents

def preprocessing(document):
    
    lemm = WordNetLemmatizer()
    stemmer = SnowballStemmer("english")
    
    processed_document = []
    
    #Tokenize
    for token in simple_preprocess(document):
        # Remove Stopwords
        if token not in STOPWORDS and len(token)>=3:
            # Stem and lemmetize
            processed_document.append(stemmer.stem(lemm.lemmatize(token, pos='v')))
    
    return processed_document

def create_bow(corporum):
    
    # Create Dictionary
    dictionary = gensim.corpora.Dictionary(corporum)
    
    # Truncation: remove words that occur in too few(15) or too many(>50%) of the documents; keep only top 100000
    dictionary.filter_extremes(no_below=1, no_above=0.5, keep_n=100000)
    
    # Create bow for each document
    bow_corpus = [dictionary.doc2bow(doc) for doc in corporum]
    
    return dictionary, bow_corpus
    

def main():
    #documents = import_data()
    
    start = timeit.time.time()
    
    with conc.ProcessPoolExecutor(4) as executor:
        processed_document = [doc for doc in executor.map(preprocessing, documents['headline_text'], chunksize=len(documents['headline_text'])//4)]
#    processed_document = documents['headline_text'].map(preprocessing)
    
    print("Preprocessing completed in {:0} sec(s)".format(timeit.time.time() - start))
    print("Showing some samples")
    
    rand_row = random.randint(0, len(documents['headline_text']))
    print("Original: " + documents['headline_text'][rand_row])
    print("Processed: " + str(preprocessing(documents['headline_text'][rand_row])))
    
    dictionary, bow_corpus = create_bow(processed_document)
    
    tf_idf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tf_idf[bow_corpus]

    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=15, id2word=dictionary, passes=2, workers=4, chunksize=len(bow_corpus)//4)
    
    for idx, topic in lda_model.print_topics(-1):
        print('Topic: {} Words: {}'.format(idx, topic))
        
    lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=15, id2word=dictionary, passes=2, workers=4, chunksize=len(corpus_tfidf)//4)
    
    for idx, topic in lda_model_tfidf.print_topics(-1):
        print('Topic: {} Word: {}'.format(idx, topic))

if __name__ == "__main__":
    main()
