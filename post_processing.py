from pre_processing import *
import seaborn as sns


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
    ax.set_yticklabels(data,fontsize=25,style='oblique')###

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
