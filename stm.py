from rpy2 import robjects #loading R inside Python
#importing functions to read pandas DF
from rpy2.robjects import pandas2ri
pandas2ri.activate()
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter

import pandas as pd

#import R base library
base = importr('base')


pd_df = pd.DataFrame({'int_values': [1,2,3],'text': ['abc', 'def', 'ghi']})
with localconverter(robjects.default_converter + pandas2ri.converter):
  r_from_pd_df = robjects.conversion.py2rpy(pd_df)


# ans = robjects.r(
#   '''
#   library(readr)
#   source('./coherence.R')
#   x = {0}
#   print({1})
#   '''.format(12,22))

robjects.r.source('stm.R')
ans = robjects.r.run_stm(pd_df,10)
print(ans)

