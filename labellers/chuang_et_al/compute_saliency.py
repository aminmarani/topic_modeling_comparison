#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import sys

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
		
		print("Computing topic info...")
		sys.stdout.flush()
		self.computeTopicInfo(model, saliency)
		
		print("Computing term info...")
		sys.stdout.flush()
		self.computeTermInfo(model, saliency)
		
		if rank:
			print("Ranking topics and terms...")
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
		for i in range(model.term_count):
			term = model.term_index[i]
			counts = model.term_topic_matrix[i]
			frequency = sum( counts )
			probs = self.getNormalized( counts )
			distinctiveness = self.getKLDivergence( probs, topic_marginal )
			saliencyVal = frequency * distinctiveness
			term_info.append( {
				'term' : term,
				'saliency' : saliencyVal,
				'frequency' : frequency,
				'distinctiveness' : distinctiveness
			} )
		saliency.term_info = term_info
	
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
