import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim import similarities

from tqdm import tqdm



import csv, sys
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


import pandas as pd
import numpy as np

import copy


try:
	csv.field_size_limit(sys.maxsize)
except:
	print('Error in setting maxSize for CSV output')



def nltk_tag_to_wordnet_tag(nltk_tag):
	'''
	Returns: The converted tag from nltk to wordnet

	parameter nltk_tag: inlcudes one of tags, adjective, Verb, Noun, or adverb	
	'''
	if nltk_tag.startswith('J'):
	    return wordnet.ADJ
	elif nltk_tag.startswith('V'):
	    return wordnet.VERB
	elif nltk_tag.startswith('N'):
	    return wordnet.NOUN
	elif nltk_tag.startswith('R'):
	    return wordnet.ADV
	else:          
	    return None


def remove_html_tags(doc:str):
    CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    return re.sub(CLEANR, '', doc).strip()

def preprocess_data(doc_list, extra_stopwords = {},len_th=4,lemmatized=False):
	'''
	Returns: a list of process dataset and origianl documents of those documents

	This function removes stop-wrods, lemmatized the documens, if stated, and eliminates the documnets 
	with lenhgth of 4 or less. 
	***These processes may result in lower number of documents than the original number. To make sure 
	you receive both the original docs and the processed doc in similar order we return both.

	parameter doc_list: a list of string (documents)
	parameter extra_stopwords: NLTK.stop_words are used, if you wish to add to that list, you can use yours.
	parameter len_th: documents with len_th and less will be removed.
	parameter lemmatized: If true, the terms will be lemmatized. **be aware that lemmatization of the documents
	will result in different topics and may need different evaluation, including NPMI, stability, or human assessment**

	'''
	orig_docs = doc_list.copy()
	# replace single smart quote with single straight quote, so as to catch stopword contractions
    #converting open/close quotations to neutral one
	doc_list = [re.sub("[\u2018\u2019]", "'", doc) for doc in doc_list]
    #removing digits
	doc_list = [re.sub('\d+', '', doc) for doc in doc_list]
	doc_list = [re.sub('(\/.*?\.[\w:]+)', '', doc) for doc in doc_list]
	#doc_list = [re.sub('pdf|icon|jpg', '', doc) for doc in doc_list]
	#doc_list = [re.sub('(http(s)?:\/\/.)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)', '', doc) for doc in doc_list]
	doc_list = [re.sub(r"http\S+", '', doc) for doc in doc_list]

	# initialize regex tokenizer
	tokenizer = RegexpTokenizer(r'[\w|#]+')#\w+
	# create English stop words list
	en_stop = set(stopwords.words('english'))
	# add any extra stopwords
	if (len(extra_stopwords) > 0):
		en_stop = en_stop.union(extra_stopwords)

	#defining a lemmatizer
	lemmatizer = WordNetLemmatizer()

	# list for tokenized documents in loop
	texts = []
	original_docs = []
	# loop through document list
	c = 0 #counter on the document number
	for i,orig in zip(doc_list,orig_docs):
		# clean and tokenize document string
		raw = i.lower()
		tokens = tokenizer.tokenize(raw)
		stopped_tokens = []
		# remove stop words from tokens
		#stopped_tokens = [i for i in tokens if not i in en_stop and len(i)>1]
		if lemmatized:
		  for t in tokens:
		    if t not in en_stop and len(t)>1:
		      pos=nltk_tag_to_wordnet_tag(nltk.pos_tag([t])[0][1])
		      if pos:
		        stopped_tokens.append(lemmatizer.lemmatize(t,pos=pos))
		      else:
		        stopped_tokens.append(lemmatizer.lemmatize(t))
		  #     print(t,pos,nltk.pos_tag([t])[0][1])
		  # print(stopped_tokens)
		  #stopped_tokens = [lemmatizer.lemmatize(i,pos=nltk_tag_to_wordnet_tag(nltk.pos_tag([i])[0][1])) for i in tokens if not i in en_stop and len(i)>1]
		else:
		  stopped_tokens = [i for i in tokens if not i in en_stop and len(i)>1]


		# add tokens to list
		if len(stopped_tokens) >= len_th:
		  texts.append(stopped_tokens)
		  original_docs.append([orig,c])

		c += 1

	return texts,original_docs


def remove_states(doc_list,stopwords):
	'''
	Returns: list of documents without state names

	**This function is deprecated**

	parameter doc_list: list of document (type: string)
	parameter stopwords: list of states or any stopwords you wish to remove from the original or processed documents
	'''
	ls = []
	for doc in doc_list:
		temp = []
		for t in doc:
		  for s in stopwords:
		    r = -1
		    try:
		      r = t.index(s)
		      if r>-1:
		        break
		    except:
		      a = 0#do nothing!
		  if r == -1:
		    temp.append(t)
		ls.append(temp)
	return ls




def prepare_corpus(doc_clean,no_below=5,no_above=0.5):
	'''
	Reutrns: A dictionary of the final set of terms and document-term frequency matrix

	# adapted from https://www.datacamp.com/community/tutorials/discovering-hidden-topics-python

	parameter doc_clean: processed set of documents (type: string)
	parameter no_below: exclude words that only appear $no_below$ times or less in the whole corpus
	parameter no_above: any words included in more than $no_above$ percentage of documents will be excluded

	'''
	# Creating the term dictionary of our courpus, where every unique term is assigned an index.
	dictionary = corpora.Dictionary(doc_clean)

	dictionary.filter_extremes(no_below=no_below, no_above=no_above)
	# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
	doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
	# generate LDA model
	return dictionary,doc_term_matrix


def loading_wiki_docs(filename:str):
	'''
	Returns: documents of wikipedia corpus

	loads a wikipedia text file and return documents with length>3

	parameter filename: name of the wikipedia text documents (type:str)
	'''
	wiki_docs = []

	with open(filename,'r',encoding="utf-8") as f:
	    d = f.readline()
	    wiki_docs.append(d)
	    while(d):
	      d = f.readline()
	      if len(d) > 3:
	        wiki_docs.append(d)
	return wiki_docs

def newsgroup(data_path):
  '''
  Read data of 20newsgroup from a CSV file and return a dataframe incluidng the actual and cleaned docs

  Returns : A pandas dataframe

  parameter data_path: path to the csv file
  '''

  text_df =  pd.read_csv(data_path,sep=';')
  #removing non-string documents/entities
  doc_list = [i for i in list(text_df.text_cleaned) if type(i) == str]
  actual_doc_list = [text_df.text[i] for i,j in enumerate(list(text_df.text_cleaned)) if type(j) == str]
  return pd.DataFrame(list(zip(doc_list,actual_doc_list)),columns=['text','text_cleaned'])

def ap_corpus(data_path):
	'''
	read data of AP corpus as one text file

	Returns : A pandas dataframe

  parameter data_path: path to the csv file
	'''
	docs = []

	with open(data_path,'r') as txtfile:
		lines = txtfile.readlines()
		for l in lines:
			if l[0] != '<':
				#it is supposed to be clean and actual text for each doc
				#but we will do pre-processing anyways; so to keep the format
				#we save it twice
				docs.append([l,l])
	return pd.DataFrame(docs,columns=['text','text_cleaned'])


def term_pairs_generator(terms):
	'''
	This function returns all the pairs in one list of terms

	returns a list of all term-pairs

	parameter terms: a list of unique terms
	'''
	term_pairs = set()

	#making pair terms
	for i in range(len(terms)):
		for j in range(i+1,len(terms)):
			term_pairs.add((terms[i],terms[j]))

	return list(term_pairs)



def transition_labeling(days_topic_df):
    '''
    Calculating if an entry is "NA", "USE", "NON-USE SHORT", or "NON-USE LONG"
    **Note that each entry should have a property names "blog".

    parameters:
    -----------------
    @param days_topic_df: dataframe including date, days, topic_dist, etc.
    
    
    returns:
    -------------
    A list of the same size as days_topic_df length with label of each row
    '''

    labels = {i:"No period" for i in list(days_topic_df.id)}
    short_delta = 10
    long_delta = 20
    
    for blog in tqdm(set(days_topic_df.blog)):
      tdf = days_topic_df[days_topic_df.blog == blog].sort_values('days')
      start = 0#starting index
    
      # for i in range(1,len(tdf)):
      i = 1#since we check index with the previous one, we start from 1
      while i<len(tdf) and start<len(tdf):
        #seeing a break or end of tdf
        if abs(tdf.iloc[i].days - tdf.iloc[i-1].days) > short_delta or i == len(tdf) -1 :
          j = i-1 #j will go back to find other periods

          nuse_inds = []
          #we check the non-use only if we are not at the end of dataframe for this blog
          if i != len(tdf)-1:
            while j >= start and abs(tdf.iloc[j].days - tdf.iloc[i-1].days)<=short_delta:
              #keeping indices as we don't know this is short or long yet!
              nuse_inds.append(tdf.iloc[j].id)
              j-=1#go back
          #meaning that we have other periods before the one we found and/or 10 days before that if it was non-use
          if j>start:
            #we have to collect use [we need this part to make sure if we reach a use or NA]
            # use = np.array([0])#storing use
            use_ls = []
            use_inds = []
            k = j
            while k>=start:
              #if the difference is more than short_delta that is a period
              if abs(tdf.iloc[k].days - tdf.iloc[j].days)>short_delta:
                #store it
                use_ls.append(1)#just adding an entry to let the algo knows we found a period
                # use = np.array([0])#storing use (a fresh start)
                j = k
              #keep track of use unless we reach to a period
              # use = np.array([1]) + use#fake assignment to keep the algo works
              use_inds.append(tdf.iloc[k].id) #storing index of use
              k-=1#go back
            #after the while-loop if use_ls has at least two periods we should save it
            if len(use_ls)>1:
              for id in use_inds:
                  labels[id] = 'use' 
    
            #we also need to collect non-use
            if i != len(tdf)-1: #if there are other records in df to check we go for checking non-use
              #checking the first next 10 days as after non-use
              j = i
              while j<len(tdf) and abs(tdf.iloc[j].days - tdf.iloc[i].days)<short_delta:
                nuse_inds.append(tdf.iloc[j].id)
                j+=1
    
              #storing short or long
              if short_delta<abs(tdf.iloc[i].days - tdf.iloc[i-1].days)<=long_delta: #short non-use
                for id in nuse_inds:
                  labels[id] = 'non-use short'
              else:#long non-use
                 for id in nuse_inds:
                     labels[id] = 'non-use long'
    
              #we need to update start as the index j we have
              start = j
            else:#if we do not explore non-use we still need to update start as
              start = i
    
    
    
        i+=1 #adding loop counter

    return labels
