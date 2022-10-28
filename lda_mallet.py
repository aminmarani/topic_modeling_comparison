import gensim


#check Gensim version
assert gensim.__version__ == '3.8.3', 'You must install Gensim 3.8.3 to be able to run LDA Mallet'


from gensim.models.wrappers import LdaMallet
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import LdaModel

import pandas as pd
import numpy as np


import math
import sys

import matplotlib.pyplot as plt

from scipy.spatial.distance import jensenshannon
try:
	from hungarian_algorithm import algorithm
except:
	print('You need to install "hungarian_algorithm" package to run this code')
	exit()


from post_processing import jaccard_sim, dice_sim, similarity_computation

import platform 
#checking OS
if 'windows' in platform.system().lower():
	mallet_path = 'c:/mallet-2.0.8/bin/mallet'
else:
	mallet_path = 'mallet'




def compute_coherence_values(dictionary, corpus, texts, limit=25, start=5, step=5,threshold=0.10,runs = 1):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    purity_values: Average purity for each run
    contrast_values: Average of contrast for each run
    df: DataFrame df inlcudes all results and number of topics associated with those results
    """
    coherence_values = []
    model_list = []
    purity_values = []
    contrast_values = []
    df = pd.DataFrame(columns=['num_topics','coherence','purity','contrast'])
    for num_topics in range(start, limit+1, step):
      model_t = []
      purity_t = []
      coherence_t = []
      contrast_t = []
      for r in range(runs):
          #model=LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
          model = LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=dictionary,optimize_interval = 25)
          model_t.append(model)

          #purity computation
          topic_term_cond = get_conditional_probabilities(model,num_topics)
          topic_term = model.get_topics()#model.show_topics(num_topics = num_topics, num_words=200)
          #go over each topic and compute set of words that satisfy p(t|w)>=threshold
          pur = []
          cont = []
          for t in range(num_topics):
            w_ind = np.argwhere(topic_term_cond[t,:]>=threshold)
            pur.append(np.sum(topic_term[t,w_ind]))
            cont.append(np.mean(topic_term_cond[t,w_ind]))
          
          #for n in range(num_topics):
            #pur.append(sum([float(i) for i in re.findall(r'\d*\.\d+',topic_term[n][1]) if float(i)>=threshold]))

          purity_t.append(np.mean(pur))
          contrast_t.append(np.mean(cont))
          coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_npmi')
          coherence_value = coherencemodel.get_coherence()
          coherence_t.append(coherence_value)
          df = df.append({'num_topics':num_topics,'coherence':coherence_value,'purity':np.mean(pur),'contrast':np.mean(cont),'coherence_std':np.std(coherencemodel.get_coherence_per_topic())},ignore_index=True)

        #stroing the results
      model_list.append(model)
      purity_values.append(purity_t)
      coherence_values.append(coherence_t)
      contrast_values.append(contrast_t)
      print('{0} number of topics has been processed'.format(num_topics))

    return model_list, coherence_values, purity_values, contrast_values,df


def get_conditional_probabilities(model,n_topics):
  #building an object with topic-term matrix, term-index, and topic-index
  r_model = {'term_topic_matrix':model.get_topics().T,'term_index':np.asarray(list(model.id2word.items()))[:,1],'topic_index':model.show_topics(num_topics = n_topics, num_words=3)}
  res = computeSaliency(r_model,True)#run the computeSaliency to get topic-term conditional probabilities

  topic_term_cond = np.zeros((n_topics,len(res.term_info)))#make an empty numpy array to store the retrieved details
  #store conditional probabilities
  for i in range(len(res.term_info)):
    topic_term_cond[:,i] = res.term_info[i]['probs']

  return topic_term_cond



class Saliency:
	pass
	
class Model:
	def __init__( self, r_model ):
		self.term_topic_matrix = r_model["term_topic_matrix"]
		self.topic_index = r_model["topic_index"]
		self.topic_count = len(self.topic_index)
		self.term_index = r_model["term_index"]
		self.term_count = len(self.term_index)

class ComputeSaliency( object ):
	"""
	Distinctiveness and saliency.
	
	Compute term distinctiveness and term saliency, based on
	the term probability distributions associated with a set of
	latent topics.
	
	Input is term-topic probability distribution, stored in 3 separate files:
		'term-topic-matrix.txt' contains the entries of the matrix.
		'term-index.txt' contains the terms corresponding to the rows of the matrix.
		'topic-index.txt' contains the topic labels corresponding to the columns of the matrix.
	
	Output is a list of term distinctiveness and saliency values,
	in two duplicate formats, a tab-delimited file and a JSON object:
		'term-info.txt'
		'term-info.json'
	
	An auxiliary output is a list topic weights (i.e., the number of
	tokens in the corpus assigned to each latent topic) in two
	duplicate formats, a tab-delimited file and a JSON object:
		'topic-info.txt'
		'topic-info.json'
	"""
		
	def execute( self, model, rank ):
		
		saliency = Saliency()
		
		#print("Computing topic info...")
		sys.stdout.flush()
		self.computeTopicInfo(model, saliency)
    
		#print("Computing term info...")
		sys.stdout.flush()
		self.computeTermInfo(model, saliency)
		
		if rank:
			#print("Ranking topics and terms...")
			sys.stdout.flush()
			self.rankResults(saliency)
		
		return saliency
	
	def computeTopicInfo( self, model, saliency ):
		topic_weights = [ sum(x) for x in zip( *model.term_topic_matrix ) ]
		topic_info = []
		for i in range(model.topic_count):
			topic_info.append( {
				'topic' : model.topic_index[i],
				'weight' : topic_weights[i]
			} )
		
		saliency.topic_info = topic_info
	
	def computeTermInfo( self, model, saliency ):
		"""Iterate over the list of terms. Compute frequency, distinctiveness, saliency."""
		
		topic_marginal = self.getNormalized( [ d['weight'] for d in saliency.topic_info ] )
		term_info = []
		#print(model.term_count)
		for i in range(model.term_count):
			term = model.term_index[i]
			counts = model.term_topic_matrix[i]
			frequency = sum( counts )
			probs = self.getNormalized(counts)
			#print(probs)
			distinctiveness = self.getKLDivergence( probs, topic_marginal )
			saliencyVal = frequency * distinctiveness
			term_info.append( {
				'probs':probs,
				# 'topic_margins':topic_marginal,
				'term' : term,
				'saliency' : saliencyVal,
				'frequency' : frequency,
				'distinctiveness' : distinctiveness
			} )
		saliency.term_info = term_info
		saliency.topic_margins = topic_marginal
	
	def getNormalized( self, counts ):
		"""Rescale a list of counts, so they represent a proper probability distribution."""
		tally = sum( counts )
		if tally == 0:
			probs = [ d for d in counts ]
		else:
			probs = [ d / tally for d in counts ]
		return probs
	
	def getKLDivergence( self, P, Q ):
		"""Compute KL-divergence from P to Q"""
		divergence = 0
		assert len(P) == len(Q)
		for i in range(len(P)):
			p = P[i]
			q = Q[i]
			assert p >= 0
			assert q >= 0
			if p > 0:
				divergence += p * math.log( p / q )
		return divergence
	
	def rankResults( self, saliency ):
		"""Sort topics by decreasing weight. Sort term frequencies by decreasing saliency."""
		ranked_term_info = sorted( saliency.term_info, key = lambda term_freq : -term_freq['saliency'] )
		for i, element in enumerate( ranked_term_info ):
			element['rank'] = i

#-------------------------------------------------------------------------------#

def computeSaliency( r_model, rank = True):
	model = Model( r_model )
	return ComputeSaliency().execute( model, rank )


def alpha_adjustment(doc_term_matrix,n_topics:int,vocab_dict,pre_processed_docs,random_seed = 54321,alpha_min=5,alpha_max=100,alpha_step=5):
	'''
	finding the alpha that maximizes coherence score

	Returns: None (plots an alpha-NPMI)

	parameter doc_term_matrix: document_term matrix (type:np_array)
	parameter n_topics: number of topics to find the best alpha for (type: int)
	parameter vocab_dict: dictionary of words (type:list)
	parameter pre_processed_docs: processed docs (type:List)

	parameter random_seed: Fixed random seed to generate similar results every time running a model with similar alpha (default:54321)
	parameter alpha_min: minimum parameter for alpha (default=5)
	parameter alpha_max: maximum parameter for alpha (default=100)
	parameter alpha_step: step value to change alpha for each run (default=5)
	'''
	coherence_value = []

	for alpha in np.arange(alpha_min,alpha_max+1,alpha_step):
		ldaMallet = LdaMallet(mallet_path, corpus=doc_term_matrix, num_topics=n_topics, id2word=vocab_dict,random_seed = random_seed,optimize_interval=alpha)
		coherencemodel = CoherenceModel(model=ldaMallet, texts=pre_processed_docs, dictionary=vocab_dict, coherence='c_npmi')
		coherence_value.append(coherencemodel.get_coherence())

	ax = plt.plot(np.arange(alpha_min,alpha_max+1,alpha_step),coherence_value)
	plt.title('Coherence score of different alpha')
	plt.show()


def get_doc_topics(model,docs,n_topics,doc_number,top_doc_n=10,show_top_doc=False):
  '''
  Computes and shows doc-topic distribution
  -----------------------------------------
  parameters:
  -----------------
  model: A LdaMallet gensim wrapper
  n_topics: number of topics that the model was trained with
  doc_number: number of documents in total
  top_doc_n: number of top documents to show for each topic
  show_top_doc: whether to show top documents for each topic or not

  returns:
  -----------------
  doc_topics_np: a numpy array of size (doc,n_topics) that shows the distribution of topics for each doc
  '''
  #test something
  doc_topics = model.load_document_topics() #loading doc-topic distribution
  doc_topics_np = np.zeros((doc_number,n_topics)) #initializing a numpy array to collect all topic-doc matrices

  docc = 0
  #reading one by one from doc_topics LDA output
  for D in doc_topics:
    doc_topics_np[docc,:] = np.asarray(D)[:,1]
    docc = docc + 1
  #top_doc_n = 10

  if show_top_doc:
    #printing top documents of each topics
    for i in range(doc_topics_np.shape[1]):
      top_doc = np.argsort(doc_topics_np[:,i])[-top_doc_n:]
      print('Topic ', i,' : ',model.show_topic(i))
      print('top docs: \n')
      for ind in reversed(range(len(top_doc))):
        print([doc_topics_np[top_doc[ind],i],docs[top_doc[ind]]])
        print("........----------------........")
      print("------------------------------------------------------------------")
      print('\n\n\n')

  return doc_topics_np





def compute_stability_values(dictionary, corpus, texts, limit=25, start=5, step=5,runs = 1,sim_funcs=[jaccard_sim,jensenshannon],func_names=['jaccard','JSD']):
    """
    Compute stability for various number of topics and with a given set of similarity function

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics
    step  : steps of changing counter for topics
    sim_functions: similarity functions

    Returns:
    -------
    df: a dataframe including all similarity values with different K and different similairy functions
    """

    coherence_values = []
    model_list = []
    purity_values = []
    contrast_values = []
    df = pd.DataFrame(columns=['num_topics'])
    for num_topics in range(start, limit+1, step):
      #storing similarity values in an array for each number of topics
      appear_sim_values = []
      prob_sim_values = []
      #list of topic terms for each run. Each entry would be a list of topic-terms
      topic_terms = []
      #list of term probabilities. Each entry would be a list of topic-term probabilities for a single run
      term_prob = []
      
      #running multiple topic models
      for r in range(runs):
          model = LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=dictionary,optimize_interval = 50)
          temp = []
          #extracting top-20 terms of each topic
          for k in range(num_topics):
            temp.append(list(np.array(model.show_topic(k,topn=20))[:,0]))
          
          #storing topic-terms and term-probability vector of the current model
          topic_terms.append(temp)
          term_prob.append(model.get_topics().T)
      
      #do the pairwise comparison
      for i in range(runs):
        for j in range(runs):
          if i==j: continue #we don't want to compute run[i] vs. run[i]
          #do pairwsie topic comparison to build a matrix matching [should be a dictionary for algorithm.matching function]
          sim_term = {}# np.zeros((len(topic_terms[i]),len(topic_terms[j])))
          sim_prob = {}# np.zeros((len(topic_terms[i]),len(topic_terms[j])))
          for t1 in range(len(topic_terms[i])):
            term_dic={} #each entry of the dictionary is also a dictionary
            prob_dic={}
            for t2 in range(len(topic_terms[j])):
              func = sim_funcs[0] #first function should be term-based similarity function
              term_dic[t2] = func(topic_terms[i][t1],topic_terms[j][t2])
              func = sim_funcs[1] #second function should be probability-based similarity function
              prob_dic[t2] = 1-func(term_prob[i][:,t1],term_prob[j][:,t2]) #Jensen and shanon is a distance function and needs to be subtract from 1
            sim_term['model1_'+str(t1)] = term_dic
            sim_prob['model1_'+str(t1)] = prob_dic
          
          #compute similairy using hungarian bi-partite matching
          try:
            appear_sim_values.append(np.mean(np.array(algorithm.find_matching(sim_term))[:,1]))
            prob_sim_values.append(np.mean(np.array(algorithm.find_matching(sim_prob))[:,1]))
          except:
            0 #an error caused by hungarian algorithm that I can't understand!
          #return np.mean(np.array(algorithm.find_matching(sim_term))[:,1]),sim_term,topic_terms,sim_prob

      #storing average similarity for each K
      df = df.append({'num_topics':num_topics,'appearance_sim_avg':np.mean(appear_sim_values),'probability_sim_avg':np.mean(prob_sim_values),'appearance_sim_std':np.std(appear_sim_values),'probability_sim_std':np.std(prob_sim_values)},ignore_index=True)

      print('All runs for {0} topics have finished'.format(num_topics))


    return df

def stability_plot(stab_df:pd.DataFrame):
	'''
	plotting the results of stabulity function for top-terms and term-topic probability

	Returns: None

	parameter stab_df: results of stability function, aka compute_stability_values as pd.DataFrame
	'''
	ax = plt.errorbar(x='num_topics',y='appearance_sim_avg',yerr='appearance_sim_std',data=stab_df)
	ax = plt.errorbar(x='num_topics',y='probability_sim_avg',yerr='probability_sim_std',data=stab_df)
	#plt.title('Coherence score with std within a single run')
	plt.legend(['appearance stability','probability stability'])
	plt.show()
