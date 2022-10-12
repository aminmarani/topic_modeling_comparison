library(reticulate)
getLabels <- function(docs, model, stop_words, label_len, delim=" "){
  print("Importing python implementation of saliency calculator")
  source_python("labellers/chuang_et_al/compute_saliency.py")
  
  print("Preparing data")
  py_model = list(term_topic_matrix = t(model$word_distr_count), topic_index = c(1:dim(model$word_distr)[1]), term_index = model$words)
  
  print("Running saliency computation")
  saliencyObj = computeSaliency(py_model, rank = F)
  saliency = sapply(saliencyObj$term_info, function(x){x[["saliency"]]})
  
  print("Building saliency dataframe")
  wordsDistr = data.frame(t(model$word_distr))
  wordsDistr["saliency"] = saliency
  rownames(wordsDistr) = model$words
  
  print("Collecting terms for labels")
  labelList = vector(mode = "character", length(model$doc_distr))
  for (i in 1:length(model$doc_distr)) {
    firstPass = head(wordsDistr[order(wordsDistr[,i], decreasing = T),], n = label_len * 5)
    labelList[i] = paste(head(rownames(firstPass[order(firstPass[["saliency"]], decreasing = T),]), n = label_len), collapse = delim)
  }
  
  return(labelList)
}