library(reticulate)
getLabels <- function(docs, model, stop_words, label_len, delim=" "){
  print("Importing python implementation of saliency calculator")
  source_python("labellers/chuang_et_al/compute_saliency.py")
  
  print("Preparing data")
  py_model = list(term_topic_matrix = t(model$word_distr_count), topic_index = c(1:dim(model$word_distr)[1]), term_index = model$words)
  
  print("Running saliency computation")
  saliencyObj = computeSaliency(py_model, rank = F)
  distinctiveness = sapply(saliencyObj$term_info, function(x){x[["distinctiveness"]]})^2
  
  print("Building dataframes")
  topicSaliency = data.frame(t(model$word_distr) * distinctiveness)
  rownames(topicSaliency) = model$words
  
  print("Collecting terms for labels")
  labelList = vector(mode = "character", length(model$doc_distr))
  for (i in 1:length(model$doc_distr)) {
    labelList[i] = paste(head(rownames(topicSaliency[order(topicSaliency[,i], decreasing = T),]), n = label_len), collapse = delim)
  }
  
  return(labelList)
}