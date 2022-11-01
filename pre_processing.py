import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim import similarities



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

	# replace single smart quote with single straight quote, so as to catch stopword contractions
	doc_list = [re.sub("[\u2018\u2019]", "'", doc) for doc in doc_list]
	doc_list = [re.sub('\d+', '', doc) for doc in doc_list]
	doc_list = [re.sub('(\/.*?\.[\w:]+)', '', doc) for doc in doc_list]
	#doc_list = [re.sub('pdf|icon|jpg', '', doc) for doc in doc_list]
	#doc_list = [re.sub('(http(s)?:\/\/.)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)', '', doc) for doc in doc_list]
	doc_list = [re.sub(r"http\S+", '', doc) for doc in doc_list]

	# initialize regex tokenizer
	tokenizer = RegexpTokenizer(r'\w+')
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
	for i in doc_list:
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
		  original_docs.append([i,c])

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
