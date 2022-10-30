from pre_processing import *
from lda_mallet import *
from post_processing import *

from os import walk

import subprocess

import random

import matplotlib.pyplot as plt
import seaborn as sns


def reading_results(res,topic_num,itreations):
  '''
  Reading result to get top terms and log-likelihood every 50 iterations

  returns: Log-Likelihood and all_top terms for each 50 iterations

  parameter res: all the command prompt prints and results (type:str)
  '''

  #reading the string
  current_line = 9#where to start in CSV file
  stp1 = topic_num+2#where to read Log-Likelihood
  stp2 = 6 #constant number of lines to get to topics again

  #please set the number of iteration to what you set for your topic model
  all_top_terms = []#storing all top terms
  LLs = []
  for _i in range(int(itreations/50)):
    #top_terms = []
    for i in range(topic_num):#reading top 
      try:
        #reading top terms splited by two tabs + the second split is for splitting the terms with space
        #excluding the last item using [0:-1].  because the last item is '\n'
        all_top_terms.extend(res[current_line].split('\t')[2].split(' ')[0:-1])
      except:
        print(res[current_line-2:current_line+3],current_line)
      current_line+=1#going to next line
    current_line+=1 #going to LL
    if _i>3: #optimizing alpha would add [beta] update after 250 iterations and we want to add one line for that
      current_line +=1
    LLs.append(float(res[current_line].split(': ')[1]))
    current_line += stp2
    #all_top_terms.append(top_terms)

  return all_top_terms,LLs

#loading reference corpus
wiki_docs = loading_wiki_docs('./data/wiki_sampled_5p.txt')
pre_processed_wiki,no_var = preprocess_data(wiki_docs)
vocab_dict, doc_term_matrix = prepare_corpus(pre_processed_wiki)

#loading the dataset we train the model on
text_df = newsgroup('./data/20newsgroup_preprocessed.csv')
doc_list = list(text_df.text_cleaned)
pre_processed_docs,filtered_docs = preprocess_data(doc_list)
vocab_dict_, doc_term_matrix_ = prepare_corpus(pre_processed_docs)

#running for one topic number
topic_num = 3
itreations = 250
iter_stp = 50#LDA stops every 50 iterations and print LLs and top terms

all_top_terms = []#storing all top terms in one vector
all_lls = [] #all of Log-Likelihood values

for _ in range(3): #three runs
  # res = subprocess.run(['python3', 'tm_run.py','--data','./data/20newsgroup_preprocessed.csv',
  #                       '--tech','lda','--num',str(topic_num),'--seed',
  #                       str(int(random.random()*100000)),'--iter',str(itreations)]
  #                       , stdout=subprocess.PIPE,stderr=subprocess.STDOUT).stdout.decode('utf-8')
  # with open('t.csv','w') as csvfile:
  #   csvfile.write(res)

  with open('t.csv','r') as csvfile:
    res = csvfile.read()
  res = res.split('\n')
  tts,LLs = reading_results(res,topic_num,itreations)
  all_lls.extend(LLs)
  all_top_terms.extend(tts)
  coherence = []

stats = pd.DataFrame(columns=['iterations','top_n','coherence','LL'])
#running all top-terms in one go!
for n in [5,10,15,20]:
  #we compute coherence scores for all topics, which is [topic_num * numnber of runs (3) * group of top terms (4 grpups: 5-10-15-20)]
  #we should compute the average for each run over all topics with same number of top terms
  cscore = CoherenceModel(topics=all_top_terms,dictionary=vocab_dict,texts=pre_processed_wiki,topn=n,coherence='c_npmi').get_coherence_per_topic()
  cscore = np.asarray(cscore).reshape(int(itreations/iter_stp),topic_num)
  coherence_avg = np.sum(cscore,axis=1)

  for it in range(itreations/iter_stp):
    stats.append(stats,pd.DataFrame(data=[(it+1)*iter_stp,n,coherence_avg[it],LL[it]],columns=['iterations','top_n','coherence','LL']),ignore_index=True)

print(stats)

ax = sns.pointplot(x='iterations',y='coherence',hue='top_n',data=stats)
ax.title('Coherence and LL with Wiki docs as ref corpus')
ax.y_label('Coherence')
ax.x_label('Iterations#')

ax2 = ax.twinx()
sns.pointplot(x='iterations',y='LL',data=stats)
ax2.y_label('Log-Likelihood')

plt.show()

print('end...')














####in the end we need to store all topic-term, topic-doc, as well as all the plots for each asessment in one frame

