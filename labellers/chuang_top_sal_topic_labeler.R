library(reticulate)
getLabels <- function(docs, model, stop_words, label_len, delim=" "){
  print("Importing python implementation of saliency calculator")
  source_python("labellers/chuang_et_al/compute_saliency.py")
  
  print("Preparing data")
  py_model = list(term_topic_matrix = t(model$word_distr_count), topic_index = c(1:dim(model$word_distr)[1]), term_index = model$words)
  
  print("Running saliency computation")
  saliencyObj = computeSaliency(py_model, rank = F)
  saliency = sapply(saliencyObj$term_info, function(x){x[["saliency"]]})
  
  print("Building dataframes")
  wordsDistr = data.frame(max.col(t(model$word_distr))) #max.col gives the column with the max value (ie, the topic with the strongest connection to that word)
  wordsDistr["words"] = model$words
  wordsDistr = wordsDistr[order(saliency, decreasing = T),]
  
  print("Collecting terms for labels")
  labelList = vector(mode = "character", length(model$doc_distr))
  for (i in 1:length(model$doc_distr)) {
    labelList[i] = paste(head(wordsDistr[wordsDistr[,1] == i,], label_len)[["words"]], collapse = delim)
  }
  
  return(labelList)
}