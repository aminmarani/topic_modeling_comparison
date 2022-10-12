import numpy as np
from scipy.sparse import issparse

import sys
from timeit import default_timer as timer
import logging
logging.basicConfig(level=logging.DEBUG)


class PMICalculator(object):
    """
    Parameter:
    -----------
    doc2word_vectorizer: object that turns list of text into doc2word matrix
        for example, sklearn.feature_extraction.test.CountVectorizer
    """
    def __init__(self, doc2word_vectorizer=None,
                 doc2label_vectorizer=None):
        self._d2w_vect = doc2word_vectorizer
        self._d2l_vect = doc2label_vectorizer

        self.index2word_ = None
        self.index2label_ = None
        
    def from_matrices(self, d2w, d2l, pseudo_count=.1):
        """
        Parameter:
        ------------
        d2w: numpy.ndarray or scipy.sparse.csr_matrix
            document-word frequency matrix
        
        d2l: numpy.ndarray or scipy.sparse.csr_matrix
            document-label frequency matrix
            type should be the same with `d2w`

        pseudo_count: float
            smoothing parameter to avoid division by zero

        Return:
        ------------
        numpy.ndarray: #word x #label
            the pmi matrix
        """     

        print("\tPMI from_matrices running...")
        sys.stdout.flush()
        start = timer()
        
        denom1 = d2w.T.sum(axis=1) #Changed from :(d2w > 0), which checks if exists in document, not how often it occurs in document
        if np.any(denom1 == 0):
            zeros = np.where(denom1 == 0)
            denom1[zeros,0] = 1
            print("\t\tWarning! Words with 0 occurrences detected: " + str(len(zeros[1])))
        
        denom2 = d2l.sum(axis=0) #Changed from :(d2l > 0)
        if np.any(denom2 == 0):
            zeros = np.where(denom2 == 0)
            denom2[zeros,0] = 1
            print("\t\tWarning! Labels with 0 occurrences detected: " + str(len(zeros[1])))
        
        sys.stdout.flush()
        
        # both are dense
        if (not issparse(d2w)) and (not issparse(d2l)):
            numer = np.matrix(d2w.T > 0) * np.matrix(d2l > 0)
            denom1 = denom1[:, None]
            denom2 = denom2[None, :]
        # both are sparse
        elif issparse(d2w) and issparse(d2l):
            numer = (d2w.T * d2l).todense()#(d2w.T > 0) * (d2l > 0)
        else:
            raise TypeError('Type inconsistency: {} and {}.\n' +
                            'They should be the same.'.format(
                                type(d2w), type(d2l)))
        
        sys.stdout.flush()
        # dtype conversion
        numer = np.asarray(numer, dtype=np.float32)
        denom1 = np.asarray(
            denom1.repeat(repeats=d2l.shape[1], axis=1),
            dtype=np.float32)
        denom2 = np.asarray(
            denom2.repeat(repeats=d2w.shape[1], axis=0),
            dtype=np.float32)

        # smoothing
        numer += pseudo_count
        
        pmi = np.log(d2w.shape[0] * (numer) / (denom1) / (denom2))
        
        end = timer()
        print("\t" + str(end - start) + " seconds to complete from_matrices portion")
        start = end
        sys.stdout.flush()

        return pmi

    def from_texts(self, tokenized_docs, labels, pseudo_count=.1):
        """
        Parameter:
        -----------
        tokenized_docs: list of list of string
            the tokenized documents

        labels: list of list of string
        
        Return:
        -----------
        numpy.ndarray: #word x #label
            the pmi matrix
        """
        print("\tPMI from_texts running...")
        
        print("\t\tRebuilding docs for d2w transform:")
        sys.stdout.flush()
        beginning = timer()
        
        docs = [" ".join(doc)
                    for doc in tokenized_docs]
        
        end = timer()
        print("\t\t" + str(end - beginning) + " seconds to complete docs rebuilding")
        start = end
        sys.stdout.flush()
        
        print("\t\td2w transform:")
        sys.stdout.flush()
        beginning = timer()
        
        #d2w = self._d2w_vect.fit_transform(docs)   #Use if not enforcing vocabulary
        d2w = self._d2w_vect.transform(docs)
        
        end = timer()
        print("\t\t" + str(end - beginning) + " seconds to complete d2w transform")
        start = end
        sys.stdout.flush()
        
        # save it to avoid re-computation
        self.d2w_ = d2w
        
        print("\t\td2l transform:")
        sys.stdout.flush()
        start = timer()
        
        d2l = self._d2l_vect.transformBigrams(tokenized_docs, labels)
        
        end = timer()
        print("\t\t" + str(end - start) + " seconds to complete d2l transform")
        start = end
        sys.stdout.flush()
        
        print("\t\tfiltering out labels with low document frequency:")
        sys.stdout.flush()
        start = timer()
        
        min_doc_freq = 3
        
        # remove the labels without any occurrences
        indices = np.asarray(((d2l > 0).sum(axis=0) > min_doc_freq).nonzero()[1]).flatten()
        d2l = d2l[:, indices]
        end = timer()
        print("\t\tnew cand label total: " + str(len(indices)))
        print("\t\t" + str(end - start) + " seconds to complete filter")
        start = end
        sys.stdout.flush()
        
        print("\t\tindex2label manipulation:")
        sys.stdout.flush()
        start = timer()
        
        indices = set(indices)
        labels = [l
                  for i, l in self._d2l_vect.index2label_.items()
                  if i in indices]
        
        self.index2label_ = {i: l
                             for i, l in enumerate(labels)}

        end = timer()
        print("\t\t" + str(end - start) + " seconds to complete index2label manipulation")
        start = end
        sys.stdout.flush()
        
        if len(self.index2label_) == 0:
            logging.warn("After label filtering, there is nothing left.")

        print("\t\tindex2word manipulation:")
        sys.stdout.flush()
        start = timer()
        
        self.index2word_ = {i: w
                            for w, i in self._d2w_vect.vocabulary_.items()}
        
        end = timer()
        print("\t\t" + str(end - start) + " seconds to complete index2word manipulation")
        start = end
        sys.stdout.flush()
        
        end = timer()
        print("\t" + str(end - beginning) + " seconds to complete from_texts portion")
        start = end
        sys.stdout.flush()
        
        return self.from_matrices(d2w, d2l, pseudo_count=pseudo_count)
