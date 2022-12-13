from pre_processing import *
from lda_mallet import *
from post_processing import *

from os import walk

import subprocess

import random
import time

import matplotlib.pyplot as plt
import seaborn as sns

from os.path import exists

import platform 
#checking OS
if 'windows' in platform.system().lower():
  python_cmd = "C:\\PROGRA~1\\Python37\\python.exe"
elif 'darwin' in platform.system().lower():
  python_cmd = 'python3'
else:
  python_cmd = 'python'

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
        # print(res[current_line-2:current_line+3],current_line)
        pass
      current_line+=1#going to next line
    current_line+=1 #going to LL
    #in the new version we remove [beta] so we don't need the lines below
    # if _i>3: #optimizing alpha would add [beta] update after 250 iterations and we want to add one line for that
      # current_line +=1
    try:
      LLs.append(float(res[current_line].split(': ')[1]))
    except:
      print(res,current_line)
    current_line += stp2
    #all_top_terms.append(top_terms)
  return all_top_terms,LLs

LL_file = 'LLs_iter_analysis.txt'
top_terms_file = 'top_terms_iter_analysis.txt'

all_top_terms = []#storing all top terms in one vector
all_lls = [] #all of Log-Likelihood values

#running for one topic number
topic_num = [30,40,50,60,70,80,90,100,120,140,160,180]
itreations = 7000
iter_stp = 50#LDA stops every 50 iterations and print LLs and top terms


#if for any reasons we have the results and just want to see the analysis
if not exists(top_terms_file) or not exists(LL_file):

  # #loading ref corpus for coherene score for lda_mallet
  # wiki_docs = loading_wiki_docs('./data/wiki_sampled_10p.txt')
  # #doing pre-processing on wiki-pedia documents
  # pre_processed_wiki, _ = preprocess_data(wiki_docs)
  # wiki_vocab_dict, _ = prepare_corpus(pre_processed_wiki)
  # del wiki_docs
  with open('./data/wiki_full_vocab.obj','rb') as objfile:
    wiki_vocab_dict = pickle.load(objfile)

  '''reading data
  '''
  # text_df = newsgroup('./data/20newsgroup_preprocessed.csv')
  # text_df = ap_corpus('./data/ap.txt')
  # doc_list = list(text_df.text_cleaned)
  # EDML corpus
  # doc_list=[]
  # with open('./data/edml.txt','r',encoding='utf-8') as txtfile:
  #   doc_list = txtfile.readlines()
  # #extra_stopwords for EDML
  # extra_stopwords = ['isnt','want','cant','wanna','im','could','ive','would','dont','get','also','us','thats','got','ur','wanted',
  #                    'may', 'the', 'just', 'can', 'think', 'damn', 'still', 'guys', 'literally', 'hopefully', 'much', 'even', 'rly', 'guess', 'anon']#anything with a length of one
  # #tweet dataset
  # doc_list=[]
  with open('./data/covid_tweets','r',encoding='utf-8') as txtfile:
    doc_list = txtfile.readlines()
  extra_stopwords_ = ['amp']

  #tokenizing
  pre_processed_docs,filtered_docs = preprocess_data(doc_list,extra_stopwords={})
  #generate vocabulary and texts
  vocab_dict, doc_term_matrix = prepare_corpus(pre_processed_docs)

  #finding stopwords that are not in Wikipedia and removing those
  extra_stopwords = set(vocab_dict.token2id.keys()).difference(set(wiki_vocab_dict.token2id.keys()))
  extra_stopwords.update(extra_stopwords_)
  pre_processed_docs,filtered_docs = preprocess_data(doc_list,extra_stopwords=extra_stopwords)
  #since we will pre-process the corpus in the tm_run.py file, we will save 
  #it in a temp file
  #vocab_dict, doc_term_matrix = prepare_corpus(pre_processed_docs)
  with open('./data/temp_corpus','w',encoding='utf-8') as txtfile:
    for t in pre_processed_docs:
      txtfile.write(' '.join(t)+'\n')



  for t_num in topic_num:
    for _ in range(3): #three runs
      res = subprocess.run([python_cmd, 'tm_run.py','--data','./data/temp_corpus',
                            '--tech','lda','--num',str(t_num),'--seed',
                            str(int(random.random()*100000)),'--iter',str(itreations),
                            '--opt_inter',str(200),'--alpha',str(80)]
                            , stdout=subprocess.PIPE,stderr=subprocess.STDOUT).stdout.decode('utf-8')
      #we have to wait till subprocess.run finishes....
      
      with open('t.csv','w') as csvfile:
        csvfile.write(res)

      # with open('t.csv','r') as csvfile:
      #   res = csvfile.readlines()
      res = res.split('\n')
      res = [i for i in res if  'beta' not in i]#removing any line with beta
      tts,LLs = reading_results(res,t_num,itreations)
      all_lls.extend(LLs)
      all_top_terms.extend(tts)

  #saving to keep in case of an Error
  with open(LL_file,'w') as txtfile:
    for tt in all_lls:
      txtfile.write(str(tt)+'\n')
  with open(top_terms_file,'w',encoding='utf-8') as txtfile:
    for tt in all_top_terms:
      txtfile.write(','.join(tt)+'\n')

  print('LDA runs are finished!')

else:#loading the pre-saved files
  #saving to keep in case of an Error
  with open(LL_file,'r') as txtfile:
    for row in txtfile:
      all_lls.append(float(row))
  with open(top_terms_file,'r') as txtfile:
    for row in txtfile:
      all_top_terms.append(row.strip().split(','))

exit()
coherence = []

stats = pd.DataFrame(columns=['K','iterations','coherence','LL'])
# #running all top-terms in one go!
# for n in [5,10,15,20]:
#   #we compute coherence scores for all topics, which is [topic_num * numnber of runs (3) * group of top terms (4 grpups: 5-10-15-20)]
#   #we should compute the average for each run over all topics with same number of top terms
#   cscore = CoherenceModel(topics=all_top_terms,dictionary=wiki_vocab_dict,texts=pre_processed_wiki,topn=n,coherence='c_npmi',processes=1).get_coherence_per_topic()
#   cscore = np.asarray(cscore).reshape(int(itreations/iter_stp)*3,topic_num) #3 : three runs
#   coherence_avg = np.mean(cscore,axis=1)
#   # print(coherence_avg)

#defining Coherence-DB scorer
scorer = lda_score(num_topics=2,alpha=10,optimize_interval=10,iterations=1000,wiki_path='./data/wiki_sampled_10p.txt',
                   db_path = './db/wiki_full',vocab_dict_path = './data/corpus.vocab', 
                   wiki_vocab_dict_path='./data/ref_corpus.vocab',npmi_skip_threshold=0.05)
# #set the wiki_docs and corpus docs parameters
scorer.wiki_vocab_dict = []
with open(scorer.wiki_vocab_dict_path,'rb') as f:
  scorer.wiki_vocab_dict = pickle.load(f)

scorer.vocab_dict = []
with open(scorer.vocab_dict_path,'rb') as f:
  scorer.vocab_dict = pickle.load(f)

c = 0 #coherence counter
itc = 0#iteration counter
for t_num in topic_num:
  for _ in range(3):#for three runs
    for it in range(int(itreations/iter_stp)):
      #set values for scorer
      scorer.all_top_terms = all_top_terms[c:c+t_num]

      #running scorer
      coherence_avg = scorer.score(None)
      stats = pd.concat([stats,pd.DataFrame(data=[[t_num,(it+1)*iter_stp,coherence_avg,all_lls[itc]]],columns=['K','iterations','coherence','LL'])],ignore_index=True)
      #adding coherence counter
      c+=t_num
      itc+=1

#save a copy
stats.to_csv('LDA_stats.csv',index=False)
print('All coherence compuation for top terms are computed')

# stats = pd.read_csv('LDA_stats.csv')

fig, axes = plt.subplots(int(len(topic_num)/2), 2, figsize=(15,25))
fig.suptitle('Iteration analysis')

K_count = 0

for i in range(0,int(len(topic_num)/2)):
    for j in range(0,2):
        ax = sns.pointplot(ax=axes[i,j],x='iterations',y='coherence',data=stats.loc[stats.K==topic_num[K_count],:])
        if i == 6:
            xlabel = 'iterations#'
        else:
            xlabel = ''
        ax.set(title='Coherence and LL with Wiki docs as ref corpus for K = {0}'.format(topic_num[K_count]),xlabel=xlabel,ylabel='Coherence')
        ax.tick_params(axis='x', rotation=90)
        ax.set(ylim=(min(stats.coherence),max(stats.coherence)))
        if j == 1:#if the plot is on right, don't print iterations for ylabel to avoid interfere
            ax.set(ylabel='')
            ax.set(yticklabels=[])
        
        ax2 = ax.twinx()
        sns.pointplot(x='iterations',y='LL',data=stats.loc[stats.K==topic_num[K_count],:],markers = '^',color='red')
        ax2.set(ylim=(min(stats.LL),max(stats.LL)))
        if j==1:#if the plot is on the right side, print log-likelihood for ylabel
            ax2.set(ylabel='Log-Likelihood')
        else:
            ax2.set(yticklabels=[])
            
        #decreasing font size for x-ticks
        ax.set_xticklabels(ax.get_xticks(), size = 5)
        
        if i<(len(topic_num)/2)-1:#remove iterations for subplots that are not at the bottom
            ax2.set(xticklabels=[])
        else:
            ax2.set(xlabel='Iterations#')
            ax2.set(xticklabels=list(range(50,iterations,50)))
            
        K_count+=1

# ax = sns.pointplot(x='iterations',y='coherence',hue='K',data=stats.loc[stats.K<11,:])
# ax.set(title='Coherence and LL with Wiki docs as ref corpus for different K',xlabel='Coherence',ylabel='Iterations#')
# ax.tick_params(axis='x', rotation=90)


# ax2 = ax.twinx()
# sns.pointplot(x='iterations',y='LL',hue='K',data=stats.loc[stats.K<11,:],markers = '^')
# ax2.set(ylabel='Log-Likelihood')

plt.show()

fig.savefig('./iteration_analysis_multiple_K.png', format='png', dpi=1200)
fig.savefig('./iteration_analysis_multiple_K.svg', format='svg', dpi=1200)

print('end...')














####in the end we need to store all topic-term, topic-doc, as well as all the plots for each asessment in one frame

