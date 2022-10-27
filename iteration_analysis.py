from pre_processing import *
from lda_mallet import *
from post_processing import *

from os import walk

import subprocess

import random


#running for one topic number
topic_num = 3
itreations = 250

for _ in range(1): #three runs
  ans = subprocess.run(['python3', 'tm_run.py','--data','./data/20newsgroup_preprocessed.csv',
                        '--tech','lda','--num',str(topic_num),'--seed',
                        str(int(random.random()*100000)),'--iter',str(itreations)]
                        , stdout=subprocess.PIPE,stderr=subprocess.STDOUT).stdout.decode('utf-8')
  #temporarily saving the results in a csv file
  with open('./data/t.csv','w') as csvfile:
    csvfile.write(ans)

print(len(ans.split('\n')))
with open('./data/t.csv','r') as csvfile:
  s = csvfile.readlines()

print(len(s))

#reading the string
start_line = 10#where to start in CSV file
stp1 = topic_num+2#where to read Log-Likelihood
stp2 = 6 #constant number of lines to get to topics again

#please set the number of iteration to what you set for your topic model
all_top_terms = []#storing all top terms
for _ in range(int(itreations/50)):
  for i in range(topic_num):#reading top terms
    None
  #loading the CSV file to store top-terms and Log-likelihood
print('end...')













####in the end we need to store all topic-term, topic-doc, as well as all the plots for each asessment in one frame

