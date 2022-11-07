from pre_processing import *
from lda_mallet import *
from post_processing import *

from os import walk

import subprocess

import random
import time

import matplotlib.pyplot as plt
import seaborn as sns

import platform 
#checking OS
if 'windows' in platform.system().lower():
  python_cmd = "C:\\PROGRA~1\\Python37\\python.exe"
elif 'darwin' in platform.system().lower():
  python_cmd = 'python3'
else:
  python_cmd = 'python.exe'

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
        all_top_terms.append(res[current_line].split('\t')[2].split(' ')[0:-1])
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

#loading ref corpus for coherene score for lda_mallet
wiki_docs = loading_wiki_docs('./data/wiki_sampled_5p.txt')
#doing pre-processing on wiki-pedia documents
pre_processed_wiki, _ = preprocess_data(wiki_docs)
wiki_vocab_dict, _ = prepare_corpus(pre_processed_wiki)
del wiki_docs

'''reading data
'''
#text_df = newsgroup('./data/20newsgroup_preprocessed.csv')
# text_df = ap_corpus('./data/ap.txt')
# doc_list = list(text_df.text_cleaned)
#EDML corpus
doc_list=[]
with open('./data/edml.txt','r',encoding='utf-8') as txtfile:
  doc_list = txtfile.readlines()
#extra_stopwords for EDML
extra_stopwords = ['isnt','want','cant','wanna','im','could','ive','would','dont','get','also','us','thats','got','ur','wanted',
                   'may', 'the', 'just', 'can', 'think', 'damn', 'still', 'guys', 'literally', 'hopefully', 'much', 'even', 'rly', 'guess', 'anon']#anything with a length of one
#tokenizing
pre_processed_docs,filtered_docs = preprocess_data(doc_list,extra_stopwords={})
#generate vocabulary and texts
vocab_dict, doc_term_matrix = prepare_corpus(pre_processed_docs)

#finding stopwords that are not in Wikipedia and removing those
extra_stopwords = set(vocab_dict.token2id.keys()).difference(set(wiki_vocab_dict.token2id.keys()))
pre_processed_docs,filtered_docs = preprocess_data(doc_list,extra_stopwords=extra_stopwords)
#since we will pre-process the corpus in the tm_run.py file, we will save 
#it in a temp file
#vocab_dict, doc_term_matrix = prepare_corpus(pre_processed_docs)
with open('./data/temp_corpus','w',encoding='utf-8') as txtfile:
  for t in pre_processed_docs:
    txtfile.write(' '.join(t)+'\n')


#running for one topic number
topic_num = 10
itreations = 4000
iter_stp = 50#LDA stops every 50 iterations and print LLs and top terms

all_top_terms = []#storing all top terms in one vector
all_lls = [] #all of Log-Likelihood values

for _ in range(3): #three runs
  res = subprocess.run([python_cmd, 'tm_run.py','--data','./data/temp_corpus',
                        '--tech','lda','--num',str(topic_num),'--seed',
                        str(int(random.random()*100000)),'--iter',str(itreations)]
                        , stdout=subprocess.PIPE,stderr=subprocess.STDOUT).stdout.decode('utf-8')
  #we have to wait till subprocess.run finishes....
  
  # with open('t.csv','w') as csvfile:
  #   csvfile.write(res)

  # with open('t.csv','r') as csvfile:
  #   res = csvfile.readlines()
  res = res.split('\n')
  tts,LLs = reading_results(res,topic_num,itreations)
  all_lls.extend(LLs)
  all_top_terms.extend(tts)

#saving to keep in case of an Error
with open('LLs_iter_analysis.txt','w') as txtfile:
  for tt in all_lls:
    txtfile.write(str(tt)+'\n')
with open('top_terms_iter_analysis.txt','w') as txtfile:
  for tt in all_top_terms:
    txtfile.write(','.join(tt)+'\n')

print('LDA runs are finished!')


coherence = []

stats = pd.DataFrame(columns=['iterations','top_n','coherence','LL'])
#running all top-terms in one go!
for n in [5,10,15,20]:
  #we compute coherence scores for all topics, which is [topic_num * numnber of runs (3) * group of top terms (4 grpups: 5-10-15-20)]
  #we should compute the average for each run over all topics with same number of top terms
  cscore = CoherenceModel(topics=all_top_terms,dictionary=wiki_vocab_dict,texts=pre_processed_wiki,topn=n,coherence='c_npmi',processes=1).get_coherence_per_topic()
  cscore = np.asarray(cscore).reshape(int(itreations/iter_stp)*3,topic_num) #3 : three runs
  coherence_avg = np.mean(cscore,axis=1)
  # print(coherence_avg)


  c =0 #coherence counter
  for _ in range(3):#for three runs
    for it in range(int(itreations/iter_stp)):
      # print([(it+1)*iter_stp,n,coherence_avg[c],all_lls[it]])
      stats = pd.concat([stats,pd.DataFrame(data=[[(it+1)*iter_stp,n,coherence_avg[c],all_lls[it]]],columns=['iterations','top_n','coherence','LL'])],ignore_index=True)
      c+=1 #adding coherence counter
  #save a copy
  stats.to_csv('LDA_stats.csv',index=False)
  print('All coherence compuation for top-{0} terms are computed'.format(n))

plt.figure(figsize=(12,12))

ax = sns.pointplot(x='iterations',y='coherence',hue='top_n',data=stats)
ax.set(title='Coherence and LL with Wiki docs as ref corpus for K={0}'.format(topic_num),xlabel='Coherence',ylabel='Iterations#')
ax.tick_params(axis='x', rotation=90)


ax2 = ax.twinx()
sns.pointplot(x='iterations',y='LL',data=stats,color='cyan')
ax2.set(ylabel='Log-Likelihood')
# ax2.y_label('Log-Likelihood')

plt.show()

print('end...')














####in the end we need to store all topic-term, topic-doc, as well as all the plots for each asessment in one frame

