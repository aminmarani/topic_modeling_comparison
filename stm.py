from rpy2 import robjects #loading R inside Python
#importing functions to read pandas DF
from rpy2.robjects import pandas2ri
pandas2ri.activate()
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter

from pre_processing import ap_corpus

import pandas as pd

import numpy as np

#import R base library
base = importr('base')


# pd_df = pd.DataFrame({'int_values': [1,2,3],'text': ['The first example is not too long'
  # , 'second example have to be short, too.', 'basically trying to write another example']})
docs = list(ap_corpus('./data/ap.txt').text)
text_df = pd.DataFrame(zip(np.arange(1,len(docs)),docs),columns=['int_values','text'])
with localconverter(robjects.default_converter + pandas2ri.converter):
  r_from_pd_df = robjects.conversion.py2rpy(text_df)


# ans = robjects.r(
#   '''
#   library(readr)
#   source('./coherence.R')
#   x = {0}
#   print({1})
#   '''.format(12,22))

robjects.r.source('stm.R')
ans = robjects.r.run_stm(text_df,topic_n=10,max_itr=50)
print(ans)

