import os
import codecs
import regex as re
import pickle

CURDIR = os.path.dirname(os.path.realpath(__file__))

def tokenize_docs(documents):
    docs = []
    for doc in documents:
        docs.append(re.findall(r"\b[\d\p{L}][\d\p{L}\\_\/&'-]+\p{L}\b",doc))
    return docs

def load_stopwords(stopwords='datasets/lemur-stopwords.txt'):
    with codecs.open(CURDIR + "/" + stopwords, 
                     'r', 'utf8') as f:
        return list(map(lambda s: s.strip(),
                   f.readlines()))
