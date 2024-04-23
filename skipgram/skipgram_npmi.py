from nltk import skipgram
from tqdm import tqdm
from math import log


class skipgram_npmi:
    def __init__(self,words,pairs,win_size,docs):
        self.word_counts = {w:0 for w in words}
        self.pair_counts = {p:0 for p in pairs}
        self.pair_weights = {p:0.0 for p in pairs}
        self.win_size = win_size
        self.docs = docs
        

    def compute_skipgram(self,s):
        '''
        computing word occurence, co-occurence, and the distance between co-occurences
        '''
        for i in range(len(s)):
            if s[i] in self.word_counts:
                # add one to the count of this term
                self.word_counts[s[i]] += 1
            for j in range(i+1,min(len(s),i+wself.in_size)):
                if (s[i],s[j]) in self.pair_counts:
                    self.pair_counts[(s[i],s[j])] += 1 #add one to co-occurences
                    self.pair_counts[(s[i],s[j])] += 1 - ( (abs(i-j) -1) / self.win_size-1)
                elif (s[j],s[i]) in self.pair_counts:
                    self.pair_counts[(s[j],s[i])] += 1 #add one to co-occurences
                    self.pair_counts[(s[j],s[i])] += 1 - ( (abs(i-j) -1) / self.win_size-1)




    def compute_npmi(self):
        for d in tqdm(self.docs):
            self.compute_skipgram(d)
        npmi = {}
        for w1,w2 in self.pair_counts.keys():
            npmi[(w1,w2)] = log(self.pair_weights[(w1,w2)] / (self.word_counts[w1]*self.word_counts[w2])) / \
                            (-1*log(self.pair_weights[(w1,w2)]))
        return npmi
    
    
def compute_skipgram_npmi(pairs,docs):
    #create the object
    ....
    #create the pairs and words
    
    #call compute_npmi
    
    #compute average for each topic
    
    #return value of each topic
        
            