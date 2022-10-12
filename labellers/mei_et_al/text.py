from scipy.sparse import (csr_matrix, lil_matrix)
from scipy import int16
from toolz.functoolz import partial
import numpy as np


class LabelCountVectorizer(object):
    """
    Count the frequency of labels in each document
    """
    
    def __init__(self):
        self.index2label_ = None
        
    def _label_frequency(self, label_tokens, context_tokens):
        """
        Calculate the frequency that the label appears
        in the context(e.g, sentence)
        
        Parameter:
        ---------------

        label_tokens: list|tuple of str
            the label tokens
        context_tokens: list|tuple of str
            the sentence tokens

        Return:
        -----------
        int: the label frequency in the sentence
        """
        label_len = len(label_tokens)
        labels = [i+1
                  for i, l in enumerate(context_tokens[:(-label_len+1)])
                  if l == label_tokens[0]]
        for j in range(1, label_len):
            labels = [i+1
                      for i in labels
                      if context_tokens[i] == label_tokens[j]]
        return len(labels)
        
    def transform(self, docs, labels):
        """
        Calculate the doc2label frequency table

        Note: docs are not tokenized and frequency is computed
            based on substring matching
        
        Parameter:
        ------------

        docs: list of list of string
            tokenized documents

        labels: list of list of string

        Return:
        -----------
        scipy.sparse.csr_matrix: #doc x #label
            the frequency table
        """
        labels = sorted(labels)
        self.index2label_ = {index: label
                             for index, label in enumerate(labels)}

        ret = lil_matrix((len(docs), len(labels)),
                         dtype=int16)
        
        for i, d in enumerate(docs):
            for j, cnt in enumerate(map(partial(self._label_frequency, context_tokens=d), labels)):
                if cnt > 0:
                    ret[i, j] = cnt
        
        return ret.tocsr()
        
    def transformBigrams(self, docs, labels):
        labels = sorted(labels)
        self.index2label_ = {index: label
                             for index, label in enumerate(labels)}

        self.label2index_ = {label: index
                             for index, label in self.index2label_.items()}
        
        ret = lil_matrix((len(docs), len(labels)),
                         dtype=int16)
        
        for d, doc in enumerate(docs):
            for word in range(len(doc) - 1):
                potentialLabel = (doc[word], doc[word+1])
                if potentialLabel in self.label2index_:
                    ret[d, self.label2index_[potentialLabel]] += 1
        return ret.tocsr()
