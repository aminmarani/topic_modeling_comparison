library(reticulate)
getLabels <- function(docs, model, stop_words, label_len){
  print("Importing python implementation of topic labeler")
  source_python("labellers/mei_et_al/label_topic.py")
  
  print("Running topic labeler")
  return(get_topic_labels(model=model$word_distr, docs=docs$text, vocabulary=model$words, stopwords=stop_words, 
                          preprocessing_steps=c('wordlen'),   #Options: 'tag', 'wordlen', 'stopword'   Use numeric() as empty list
                          n_cand_labels=2000, label_min_df=5, labels_per_topic=label_len, 
                          label_tags=c('NN,NN', 'JJ,NN')))    #label_tags is ignored if 'tag' isn't in preprocessing_steps
}