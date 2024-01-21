from pre_processing import *
import seaborn as sns
import copy
from matplotlib import pyplot as py

from tqdm import tqdm

import pandas as pd





def jaccard_sim(ls1,ls2):
  #gets two list and returns Jaccard value
  set1 = set(ls1); set2 = set(ls2)
  return len(set1.intersection(set2))/len(set1.union(set2))

def dice_sim(ls1,ls2):
  #gets two list and returns Jaccard value
  set1 = set(ls1); set2 = set(ls2)
  return (2*len(set1.intersection(set2)))/(len(set1) + len(set2))


def similarity_computation(topics1,topics2,sim_func=dice_sim):
  """
  compute topic similarity of two runs

  parameters:
  -----------------
  list (list of lists) of terms (number_topics * number_terms)
  list (list of lists) of terms (number_topics * number_terms)
  similarity function : dice or Jaccard (Dice is the default function)


  returns:
  -------------
  numpy 2-d arrays of terms (number_topics in topics1 * number_topics in topics2)
    """


  sim_matrix = np.zeros((len(topics1),len(topics2)))

  for i in range(len(topics1)):
    for j in range(len(topics2)):
      sim_matrix[i,j] = sim_func(topics1[i],topics2[j])

  return sim_matrix


def topic_doc_dist_threshold(records,n_topics,group,start_col=11,bins=6):
  """
  Computes topic assignment of each document w.r.t. to a threshold and returns 
  average of topics distributions for a column (e.g., blog name)

  parameters:
  -----------
  records: the dataframe of the data
  n_topics: number of topics
  start_col: The column that starts as topic = 0 

  returns:
  -----------
  average of topic distribution for the selected group
  """
  topic_dist = pd.DataFrame(records.iloc[:,start_col:])

  #article to pick the best bin-wdith and way to compute it (link from stat exchange)
  #https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
  #https://stats.stackexchange.com/questions/798/calculating-optimal-number-of-bins-in-a-histogram
  # q75,q25 = np.percentile(topic_dist.iloc[:,10],[75,25])
  # iqr = q75 - q25
  # h = 2 * iqr * len(topic_dist)**(1/3)
  # print(1/h)

  #find threshold for each topic 
  #I set the threshold one a bin has less than half of population than the previous bin
  topic_thresholds = []
  for i in range(0,n_topics):
    h,b = np.histogram(topic_dist.iloc[:,i],bins=bins)
    for j in range(1,len(h)):
      if abs(h[j]-h[j-1]) > h[j]:
        topic_thresholds.append(b[j-1]*1.5)
        topic_dist.iloc[topic_dist.iloc[:,i] <b[j-1]*1.5,i] = 0 #set any document-topic dist that are below threshold to zero
        #print(sum(h[j:]))
        break


  records_topic_avg = pd.DataFrame(records)
  #replacing obtained topic-dist from previous cell
  records_topic_avg.iloc[:,start_col:] = topic_dist
  #averaging over group
  records_topic_avg = records_topic_avg.groupby(group).sum()
  #sum over each topic, so later we can compute state-topic proportion that sums up to 1
  topic_sum = np.sum(topic_dist)#sum of doc-topic for each topic after removing documents with doc-topic below threshold
  #states summation 
  #dividing states summations by topic_sum to obtain satet proportion
  #records_topic_avg = pd.DataFrame(records_topic_avg.values/np.reshape(topic_sum.values,(1,35)))
  # print((records_topic_avg.values).shape)
  # print(np.reshape(topic_sum.values,(1,n_topics)).shape)
  records_topic_avg.iloc[:,:] = (records_topic_avg.values/np.reshape(topic_sum.values,(1,n_topics)))
  records_topic_avg

  return records_topic_avg


import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib

def heatmap(data, ax=None,title_text="A plot",
            cbar_kw={}, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom",fontsize=35)
    cbar.set_label(label='Percentage Values (X100)',weight='bold',fontsize=25)
    cbar.ax.tick_params(labelsize='large')

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))

    # ... and label them with the respective list entries.
    ax.set_xticklabels(data,fontsize=25,style='oblique')###
    ax.set_yticklabels(data.index,fontsize=25,style='oblique')###

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    #ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.set_yticklabels(data.index.values)
    #ax.set_yticks(np.arange(data.shape[1]+1)-.5,records_state_avg.index.values)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=5)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title(title_text,fontsize=40)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.4f}",textcolors=["white", "black"],threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()/2)

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def topic_selection(mat):
    """
    A function to select top topics for each document. 
    It uses the Elbow method to select top topics for each document.

    Parameters
    ----------
    mat
        the topic-doc matrix in size of (documents*topics)

    Returns
        list of topics for each documnet
        binary list of topics for each document. 1 means selected and 0 means not selected
    """
    topics4docs = [] #storing topic selection for each document
    topics4docs_bin = np.zeros((mat.shape)) #storing topic selection for each document as binary
    for i in range(mat.shape[0]):#do it for every document
        #sorting topic-doc distribution from highest to lowest
        args = np.argsort(mat[i])[::-1]
        #calculating the difference between each two topic distribution from highest to lowest but consecutievly
        difs = [abs( mat[i,args[a]] - mat[i,args[a+1]]) for a in range(len(args)-1)]
        #find the max difference
        idx = np.argmax(difs)
        #storing top topics for this doc
        topics4docs.append(args[0:idx+1])
        topics4docs_bin[i,args[0:idx+1]] = 1
    return topics4docs,topics4docs_bin


import matplotlib.pyplot as plt

def topic_author_plots(csv_file):
  '''
  Returns: None

  plots the output of topic_author Thmpson and Mino analysis

  parameter csv_file: the outpout of topic author entropy analysis
  '''
  topic_author = pd.read_csv(csv_file,sep='\t')

  counts, bins = np.histogram(topic_author['Author Entropy'])
  plt.hist(bins[:-1], bins, weights=counts)
  plt.axvline(x=topic_author['Author Entropy'].median(),linestyle='--',color='black')
  plt.title('Author Entropy')
  plt.plot()

  plt.figure()
  counts, bins = np.histogram(topic_author['Minus Major Author'])
  plt.hist(bins[:-1], bins, weights=counts)
  plt.axvline(x=topic_author['Minus Major Author'].median(),linestyle='--',color='black')
  plt.title('Minus Major Author')
  plt.plot()

  plt.figure()
  counts, bins = np.histogram(topic_author['Balanced Authors'])
  plt.hist(bins[:-1], bins, weights=counts)
  plt.axvline(x=topic_author['Balanced Authors'].median(),linestyle='--',color='black')
  plt.title('Balanced Authors')
  plt.plot()


def prepare_authorless_data(df,lda_model,pre_processed_docs):
  '''
  Returns: None

  Prepares the data of LDA mallet for authorless post-processing

  parameter df: A Pandas DataFrame including blogname and text
  parameter lda_model: Gensim LDA model
  parameter pre_processed_docs: processed documents
  '''
  #reindexing current dataframe to store corpus
  t = pd.DataFrame(df[['blog_name','text']])
  t.index = np.arange(0,len(t))
  #replacing '\n' with ' ' 
  t.text = t.text.str.replace('\n',' ').str.lower()

  #first we need to remove terms that are not within LDAMallet vocabulary

  #1. getting LDAMallet vocabulary
  vocab_ls = pd.DataFrame.from_dict(data=lda_model.id2word,orient='index').values[:,0]

  #2. remove terms outside the vocabulary
  for i in range(len(pre_processed_docs)):
    temp = []
    for s in pre_processed_docs[i]:
      if s in vocab_ls:
        temp.append(s)
    pre_processed_docs[i] = temp

  #3. replacing the new editted texts with 'text' column of the dataframe
  t.text = [' '.join(s) for s in pre_processed_docs]

  #4. removing rows with no remaining terms
  t = t[t.text != '']

  #saving dataframe as a csv file. Later this is going to be used for author-topic plots
  t.to_csv('corpus',sep='\t',encoding="UTF-8",header=False,index=True)
  #saving LDAMallet model to compute author-topic correlations
  ldaMallet.save('ldaMallet_model.gensim')




def plotting_coherence(eval_df):
  '''
  plotting coherence and std for different number of topics

  Returns: None

  parameter eval_df: Pandas DataFrame of the multiple coherene score for multiple runs for each topic number
  '''
  ax = sns.pointplot(x='num_topics',y='coherence',data=eval_df)
  plt.title('Coherence score with Wiki docs as ref corpus')
  plt.show()

  ax = plt.errorbar(x='num_topics',y='coherence',yerr='coherence_std',data=eval_df[0:-1:3])
  plt.title('Coherence score with std within a single run')
  plt.show()


def topic_top_terms(ldaMallet,n_topics,top_n_terms = 20):
  '''
  gets an LDAMallet model and returns top-terms in a list

  Returns: list of top terms (type:List)

  parameter ldaMallet: Trained Gensim Object for ldaMallet
  '''


  topic_term = []#np.asarray([['sample_string']*top_n]*n_topics)

  # loop through all the topics we have
  for i in range(n_topics):
    text_to_show = ''
    temp_ls = []
    # looping through the number of words we want to represent each topic ==> can do it with an iterator oved show_topic results as well.
    for j in range(top_n_terms): 
      temp_ls.append(ldaMallet.show_topic(i,topn=top_n_terms)[j][0])
    topic_term.append(temp_ls[:])

  return topic_term

def col_proportion(topic_avg,df,col=9):
  x = pd.DataFrame(df.iloc[:,9].value_counts()).sort_index().values / np.sum(pd.DataFrame(df.iloc[:,9].value_counts()).sort_index().values)
  # print(x)
  return topic_avg * x


def topic_author_heat_map(doc_topics,topic_term,sel_df,author_names='blog_name',start_col=10):
  '''
  plot topic_author_heat map

  returns: None

  parameter doc_topics: document-topic distribution (type:np.array)
  parameter topic_term: top terms of all topics (type: list)
  parameter author_names: names of the authors in general (default='blog names')
  parameter start_col: columns that topic-doc distribution starts from (default=10)
  '''

  #merging doc-topics distribution with selected_doc dataframe
  tdf = pd.DataFrame(doc_topics)
  tdf.index = sel_df.index
  sel_df = sel_df.join(tdf)

  n_topics = len(topic_term) #number of topics

  #extracting topic labels to use for column labels for each topic
  topic_labels = ['_'.join(i) for i in np.array(topic_term)[:,0:3]]
  col_names = dict(zip(list(np.arange(0,n_topics)),topic_labels))
  sel_df = sel_df.rename(columns=col_names)

  #getting avg of topic distribution for each group (blog name here)
  t = copy.deepcopy(sel_df)
  topic_avg_df = topic_doc_dist_threshold(t,n_topics=n_topics,group=author_names,start_col=start_col)

  #computing topic proportion
  topic_avg_proportion_df = col_proportion(topic_avg_df,sel_df,col=start_col-1)

  #visualizing the final results
  fig, ax = plt.subplots(figsize=(60,15))

  im, cbar = heatmap(topic_avg_df,ax=ax, cmap="inferno",title_text='Doc-Topic dist average on Topics')
  texts = annotate_heatmap(im, valfmt="{x:.4f}",fontsize=10,weight="bold",)
  fig.tight_layout()
  plt.show()




def date2days(time_df):
    """
    Compute days of a post from the time a blogger start posting. E.g., 10 means 10 days since they posted for the first time
    
    parameters:
    -----------------
    @param time_df: dataframe including date in dateformat and blog
    
    
    returns:
    -------------
    dataframe with an additional columns as "days"
    """
    days_df = pd.DataFrame()
    
    for blog in set(time_df.blog):
      tdf = time_df[time_df.blog==blog].sort_values('date')
      tdf['days'] = [abs((tdf.iloc[0].date - tdf.iloc[i].date).days) for i in range(len(tdf))]
      days_df = pd.concat([days_df,tdf])
    
    return days_df



def transition_analysis(days_topic_df,n_topics,begin_col,end_col):
    '''
    Computing Transitions 
    **important note for each tuple item: i[1] ==> i[0] meaning that index=1 is b4 and i[0] is a4**

    parameters:
    -----------------
    @param days_topic_df: dataframe including date, days, topic_dist, etc.
    @param n_topics: number of topics
    @param begin_col: the column topic_dist starts
    @param end_col: the column that topic_dist ends or begin_col+n_topics
    
    
    returns:
    -------------
    Three arrays in a dictionary for "use", "short non-use", and "long non-use"
    '''

    use_trs = []
    short_trs = []
    long_trs = []
    # begin_col = 10; end_col = 20
    # Topic0_dist, Topici_select
    short_delta = 10
    long_delta = 20
    
    for blog in tqdm(set(days_topic_df.blog)):
      tdf = days_topic_df[days_topic_df.blog == blog].sort_values('days')
      start = 0#starting index
    
      # for i in range(1,len(tdf)):
      i = 1#since we check index with the previous one, we start from 1
      while i<len(tdf) and start<len(tdf):
        #seeing a break or end of tdf
        if abs(tdf.iloc[i].days - tdf.iloc[i-1].days) > short_delta or i == len(tdf) -1 :
          j = i-1 #j will go back to find other periods
    
          #we check the non-use only if we are not at the end of dataframe for this blog
          if i != len(tdf)-1:
            non_use_start = np.array([0]*n_topics)#storing b4 non-use (10 days beofore index i-1)
            while j >= start and abs(tdf.iloc[j].days - tdf.iloc[i-1].days)<=short_delta:
              #we multiply topic_select to topic_dist to only keep the selected ones proportion
              non_use_start = (tdf.iloc[j,begin_col:end_col].values) * \
                                (tdf.iloc[j,end_col:end_col+n_topics].values) + non_use_start
              j-=1#go back
          #meaning that we have other periods before the one we found and/or 10 days before that if it was non-use
          if j>start:
            #we have to collect use
            use = np.array([0]*n_topics)#storing use
            use_ls = []
            k = j
            while k>=start:
              #if the difference is more than short_delta that is a period
              if abs(tdf.iloc[k].days - tdf.iloc[j].days)>short_delta:
                #store it
                use_ls.append(use.copy())
                use = np.array([0]*n_topics)#storing use (a fresh start)
                j = k
              #keep track of use unless we reach to a period
              use = tdf.iloc[k,begin_col:end_col].values * \
                    (tdf.iloc[k,end_col:end_col+n_topics].values)+ use
    
              k-=1#go back
            #after the while-loop if use_ls has at least two periods we should save it
            if len(use_ls)>1:
              for l in range(0,len(use_ls)-1):
                #we save them in reverse order, so we store them as before and after, i+1 and i respectively
                use_trs.append([use_ls[l+1], use_ls[l]])
    
            #we also need to collect non-use
            if i != len(tdf)-1: #if there are other records in df to check we go for checking non-use
              #checking the first next 10 days as after non-use
              j = i
              non_use_end = np.array([0]*n_topics)#storing b4 non-use (10 days beofore index i)
              while j<len(tdf) and abs(tdf.iloc[j].days - tdf.iloc[i].days)<short_delta:
                non_use_end = tdf.iloc[j,begin_col:end_col].values * \
                             (tdf.iloc[j,end_col:end_col+n_topics].values) + non_use_end
                j+=1
    
              #storing short or long
              if short_delta<abs(tdf.iloc[i].days - tdf.iloc[i-1].days)<=long_delta: #short non-use
                short_trs.append([non_use_start,non_use_end])
              else:#long non-use
                long_trs.append([non_use_start,non_use_end])
    
              #we need to update start as the index j we have
              start = j
            else:#if we do not explore non-use we still need to update start as
              start = i
    
    
    
        i+=1 #adding loop counter

    return {'use':use_trs,'short_non_use':short_trs,'long_non_use':long_trs}

def trns2mat(trns):
    '''
    Computing transition matrix

    parameters:
    -----------------
    @param trns: list of np arrays including a4,b4 tuples for periods
    
    
    returns:
    -------------
    Returns a np matrix of n_topics*n_topics size showing the transition from rows to columns
    '''
    n_topics = len(trns[0][0])
    mat = np.zeros((n_topics,n_topics))
    for a4,b4 in trns:
        t = b4.reshape(1,n_topics) * a4.reshape(n_topics,1)
        mat += np.float64(t)

    return mat


def topic_terms_bert(model,K):
    '''
    storing term probabilities for each topic in BERTopic

    parameters:
    -----------------
    @param model: Object of BERTopic model
    @param K: number of topics
    
    
    returns:
    -------------
    Returns a list of lists
    '''
    res = []
    for k in range(K):
        res.append([j[1] for j in model.get_topic(k)])
    return res