getLabels <- function(docs, model, stop_words, label_len, delim=" "){
  wordsDistr = data.frame(t(model$word_distr))
  rownames(wordsDistr) = model$words
  
  labelList = rep(NA, length(model$doc_distr))
  for (i in 1:length(model$doc_distr)) {
    labelList[i] = paste(head(rownames(wordsDistr[order(wordsDistr[[i]], decreasing = T),]), n = label_len), collapse = delim)
  }
  
  return(labelList)
}