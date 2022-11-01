from pre_processing import *
from lda_mallet import *
from post_processing import *

from os import walk


#reading data for ED corpus
# datafolder = './data/ed_recovery_formatted/Excel'
# #datafolder = 'ed_recovery_topicmodel'
# df = pd.DataFrame(columns=['url','type','photo','date','tags','notes','text','photo_url','reblogged','blog_name'])

# for dirpath,dirnames,filenames in walk(datafolder):
#   for filename in filenames:
#     if filename.endswith('.xlsx'):
#       t = pd.read_excel(datafolder+'/'+filename,names=['url','type','photo','date','tags','notes','text','photo_url','reblogged'])
#       blog_name = t.iloc[0,0].split(':')[1]
#       t['blog_name'] = blog_name
#       df = df.append(t.iloc[3:,:],ignore_index=True)
#       print('blog:{0}   with posts:{1}    and reblogs:{2}  '.format(filename,len(t),len(t[t.reblogged=='yes'])))


# print('number of blogs: {0} - number of posts: {1}'.format(len(set(df.blog_name)),len(df)))
# print('out of {0} documents, {1} are reblogged.'.format(len(df),len(df[df.reblogged == 'yes'])))

# #finding reblogged texts
# texts = sorted(df.text) #sort them to keep smallest post (perhaps original one) at first
# re_texts = []

# while len(texts):
#   t = [texts.pop(0)]#pop first text and find it!
#   if t[0] == ' ' or len(t[0].split())<3: 
#     continue #almost nothing to look
#   i = 0
#   while i<len(texts):
#     if t[0] in texts[i]:
#       t.append(texts.pop(i))
#     else:
#       i += 1
#   if len(t) > 1:
#     re_texts.append(t)


# print('number of unique reblogged texts: {0}'.format(len(re_texts)))
# print('number of unique string in all texts: {0}'.format(len(set(df.text))))

# extra_stopwords = ['isnt','want','cant','wanna','im','could','ive','would','dont','get','also','us','thats','got','ur','wanted',
#                    'may', 'the', 'just', 'can', 'think', 'damn', 'still', 'guys', 'literally', 'hopefully', 'much', 'even', 'rly', 'guess', 'anon']#anything with a length of one
                   

# '''pre-processing'''
# # original_doc_set = list(df.text[df.photo=='no'])
# sel_df = df[df.photo=='no'] #extracting only-text posts
# original_doc_set = list(sel_df.text)
# pre_processed_docs,filtered_docs = preprocess_data(original_doc_set,extra_stopwords=extra_stopwords)
# vocab_dict, doc_term_matrix = prepare_corpus(pre_processed_docs)
# ind = [int(i) for i in np.array(filtered_docs)[:,1]]
# sel_df = sel_df.iloc[ind,:]
# print('size of orginal dataset: {0} and size of the pre-processed dataset: {1}'.format(len(original_doc_set),len(pre_processed_docs)))

#lemmatization
# pre_processed_docs_lem,filtered_docs_lem = preprocess_data(original_doc_set,extra_stopwords=extra_stopwords,lemmatized=True)
# vocab_dict_lem, doc_term_matrix_lem = prepare_corpus(pre_processed_docs_lem)

# print('Vocab size before lemmatiziation: {0} and after lemmatization: {1}'.format(len(vocab_dict),len(vocab_dict_lem)))


'''reading data for 20newsgroup
'''
#text_df = newsgroup('./data/20newsgroup_preprocessed.csv')
text_df = ap_corpus('./data/ap.txt')
doc_list = list(text_df.text_cleaned)
#tokenizing
pre_processed_docs,filtered_docs = preprocess_data(doc_list)
#generate vocabulary and texts
vocab_dict, doc_term_matrix = prepare_corpus(pre_processed_docs)

#coherene score for lda_mallet
wiki_docs = loading_wiki_docs('./data/wiki_sampled_5p.txt')
#doing pre-processing on wiki-pedia documents
pre_processed_wiki,no_var = preprocess_data(wiki_docs)

no_var = []

lim = 151
st = 50
stp = 50
models, coherence, pur, cont,eval_df = compute_coherence_values(dictionary=vocab_dict, corpus=doc_term_matrix, texts=pre_processed_wiki, limit=lim, start=st, step=stp,threshold=0.10,runs = 3)
#running on a VM machine
eval_df.to_csv('coherence_ap_50to150.csv',index=False)
plotting_coherence(eval_df)


#Coherence Evaluation for different optimize interval values
#alpha_adjustment(doc_term_matrix=doc_term_matrix,n_topics=25,vocab_dict=vocab_dict,pre_processed_docs=pre_processed_docs,alpha_min=1,alpha_max=101,alpha_step=20)

#comparing stability value for LDA
# stab_df = compute_stability_values(dictionary=vocab_dict, corpus=doc_term_matrix, texts=pre_processed_docs, limit=10, start=6, step=1,runs = 2)
# stability_plot(stab_df)

#running topic modeling with selected parameters
# n_topics = 6; iterations = 1000
# ldaMallet = LdaMallet(mallet_path, corpus=doc_term_matrix, num_topics=n_topics, id2word=vocab_dict,iterations=iterations,random_seed = 54321,optimize_interval=50)


# #plotting author-topic heatmap
# doc_topics = get_doc_topics(ldaMallet,sel_df.text.values,n_topics,len(doc_term_matrix),top_doc_n=10,show_top_doc=False)
# topic_term = topic_top_terms(ldaMallet,n_topics)
# topic_author_heat_map(doc_topics,topic_term,sel_df)
















####in the end we need to store all topic-term, topic-doc, as well as all the plots for each asessment in one frame

