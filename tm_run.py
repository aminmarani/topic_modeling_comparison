from pre_processing import *
from lda_mallet import *
from post_processing import *

from os import walk

import subprocess
import sys, getopt





if __name__ == '__main__':
  print('************')
#  ans = subprocess.run(['python3', 'iteration_analysis.py'], stdout=subprocess.PIPE,stderr=subprocess.STDOUT).stdout.decode('utf-8')
  try:
    data_path = ''
    technique = ''
    num_topics = 100
    random_seed = 12345
    iterations = 900
    optimize_interval = 50
    alpha = 100

    opts,args = getopt.getopt(sys.argv[1:],"h",['data=','tech=','num=','seed=','iter=','opt_inter=','alpha='])

  except getopt.GetoptError:
    print('python3 tm_run.py --data data_path --tech topic_modeling_technique --num number_of_topics --seed random_seed --iter iterations --opt_inter optimization_inteval --alpha alpha')

  for opt,arg in opts:
    if opt == '-h':
      print('python3 tm_run.py --data data_path --tech topic_modeling_technique --num number_of_topics --seed random_seed --iter iterations')
    elif opt == '--data':
      data_path = arg
    elif opt == '--tech':
      technique = arg.lower()
    elif opt == '--num':
      num_topics = int(arg)
    elif opt == '--seed':
      random_seed = int(arg)
    elif opt == '--iter':
      iterations = int(arg)
    elif opt == '--opt_inter':
      optimize_interval = int(arg)
    elif opt == '--alpha':
      alpha = int(arg)


  #loading data
  if '20newsgroup' in data_path:
    text_df = newsgroup(data_path)
  if 'ap' in data_path:
    text_df = ap_corpus(data_path)
  if 'temp' in data_path:
    docs = []
    with open(data_path,'r',encoding='utf-8') as txtfile:
      docs = txtfile.readlines()
    text_df = pd.DataFrame(docs,columns=['text_cleaned'])

  doc_list = list(text_df.text_cleaned)
  #tokenizing
  pre_processed_docs,filtered_docs = preprocess_data(doc_list)
  #generate vocabulary and texts
  vocab_dict, doc_term_matrix = prepare_corpus(pre_processed_docs)
  #run topic modeling technique
  if technique == 'lda':
    ldaMallet = LdaMallet(mallet_path, corpus=doc_term_matrix, num_topics=num_topics, id2word=vocab_dict,iterations=iterations,random_seed = random_seed,optimize_interval=optimize_interval,alpha=alpha,workers=4)

















####in the end we need to store all topic-term, topic-doc, as well as all the plots for each asessment in one frame

