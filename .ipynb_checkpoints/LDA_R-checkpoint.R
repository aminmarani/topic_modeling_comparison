#checking installed package to warn users
if (! require('rJava'))
	stop('rJava package is not installed! You need to isntall this package to continue. \n **You need Java to install this package**')

if (! require('mallet'))
	stop('mallet package is not installed! You need to isntall this package to continue.')

if (! require('reshape2'))
	stop('reshape2 package is not installed! You need to isntall this package to continue.')	

if (! require('reticulate'))
	stop('reticulate package is not installed! You need to isntall this package to continue.')	

if (! require('qdapTools'))
	stop('qdapTools package is not installed! You need to isntall this package to continue.')	

if (! require('reader'))
	stop('reader package is not installed! You need to isntall this package to continue.')	

if (! require('dplyr'))
	stop('dplyr package is not installed! You need to isntall this package to continue.')	

if (! require('ggplot2'))
	stop('ggplot2 package is not installed! You need to isntall this package to continue.')	

if (! require('jsonlite'))
	stop('jsonlite package is not installed! You need to isntall this package to continue.')	




# #loading libraries
library(dplyr)
library (rJava)
.jinit(parameters="-Xmx12g") #Give rJava enough memory
library(mallet)
library(ggplot2)
library(reshape2)
library(jsonlite)
library(reticulate)



#' Runs topic-modeling, LDA Mallet, using R
#' @param docs documents with text and other meta-data
#' @param n_topics number of topics
#' @param rand_seed random seed for topic initialization
#' @param burnin_iteration update alpha hyper-parameter each burnin_iterations
#' @param after_iteration_burnin start updating alpha hyper-parameter each after_iteration_burnin
#' @param epochs training epochs
#' @param extra_epochs running model for another extra_epochs to generate final topics
#' @param word.freq words with frequency lower than this will be filter after topic generation
#' @param label.len Top-N terms to return
#' @param save_flag if you want to store the model at the end (default = False)
#' @param save_path the address to save the model, if save_flag = T

findTopics <- function(docs, n_topics,rand_seed=54321L,burnin_iteration=20,after_iteration_burnin = 10,
                      epochs=2000,extra_epochs = 50,word_min_freq=2,label.len=50,save_flag = F,
                       save_path = 'default' ){
  #Use mallet as in sample code
  print("Building mallet instance ...")
  # replace single smart quote with single straight quote, so as to catch stopword contractions
  #docs$text <- gsub("[\u2018\u2019]", "'", docs$text)
  mallet.instances <- mallet.import(docs$title, docs$text,token.regexp = "[\\p{L}|\\#]+")
    
  
  ## Create a topic trainer object.
  print("Building topic trainer ...")
  topic.model <- MalletLDA(num.topics=as.numeric(n_topics))
  topic.model$setRandomSeed(rand_seed)
  
  ## Load our documents. We could also pass in the filename of a 
  ##  saved instance list file that we build from the command-line tools.
  print("Loading documents in mallet instance into topic trainer ...")
  topic.model$loadDocuments(mallet.instances)
  
  ## Optimize hyperparameters every 20 iterations, 
  ##  after 10 burn-in iterations.
  topic.model$setAlphaOptimization(as.numeric(burnin_iteration), as.numeric(after_iteration_burnin))
  
  ## Now train a model.
  ##  We can specify the number of iterations. Here we'll use a large-ish round number.
  print("Training model ...")
  topic.model$train(as.numeric(epochs))#1000
  topic.model$maximize(as.numeric(extra_epochs))#50
  print("Training complete.")
  
  doc.topics <- data.frame(mallet.doc.topics(topic.model, smoothed=T, normalized=T)) #docs * topics matrix (topic makeup of documents)
  topic.words.count <- mallet.topic.words(topic.model, smoothed=T, normalized=F) #topics * words (word distribution over topics)
  
  word.freq <- mallet.word.freqs(topic.model)
  word.freq$term.freq <- colSums(mallet.topic.words(topic.model, normalized=F), na.rm=TRUE) #fixes issue with term count but not doc count
  #gm
  word.freq.filt <- word.freq
  word.freq.filt$id <- 1:nrow(word.freq.filt)
  word.freq.filt <- filter(word.freq.filt, term.freq >= word_min_freq) #Filters out low frequency words
  word.freq <- word.freq[word.freq.filt$id,] #Removes word from word.freq
  
  topic.words.count <- topic.words.count[,word.freq.filt$id] #Removes columns of words filtered out
  topic.words <- sweep(topic.words.count, 1, rowSums(topic.words.count), `/`) #Normalizes topic distribution
  
  wordlist <- as.character(word.freq.filt[,1]) #List of words in document. Order matchs order of words in topic.words
  
  model = list(doc_distr = doc.topics, word_distr = topic.words, words = wordlist, 
            malletModel = topic.model, wordFreq = word.freq, word_distr_count = topic.words.count) #malletModel given for easy implementation of topn labeler 
    
  if (save_flag)
      {
        if (save_path == 'default')#change the path
            save_path = paste('K',n_topics,'burnin_iteration',burnin_iteration,
                             'after.burnin',after_iteration_burnin,'epochs',epochs,
                              Sys.Date(),Sys.time(),sep='_')
        # save(topic.model, ascii=FALSE, file=paste('MalletModel_',save_path))
        save.mallet.instances(mallet.instances,file=paste('MalletInstance_',save_path))
        save.mallet.state(topic.model,state.file=paste('MalletState_',save_path,'.gz'))
        save(model, ascii=FALSE, file=paste('MalletSpec_',save_path))
        ################Important##############
        #in order to load LDA MALLET topic.model you should do
        #1. make an object ==>   topic.model <- MalletLDA(num.topics='any number is fine. Next line will override this with the actual number you stored your model with.')
        #2. load stored states ==> load.mallet.state(topic.model,state.file=your_file)
      }
                                                                                                   #but it is only saved for the session
  return(list(model,getLabels(docs,model,label.len,delim=" ")))
}


#' Getting top N-terms as labels for each topic
#' @param docs documents with text and other meta-data
#' @param model the LDA model after generating topics
#' @param label_len N for top-N terms
#' @param delim the delimiter between top-N terms

getLabels <- function(docs, model, label.len=20, delim=" "){
  wordsDistr = data.frame(t(model$word_distr))
  rownames(wordsDistr) = model$words
  
  labelList = rep(NA, length(model$doc_distr))
  for (i in 1:length(model$doc_distr)) {
    labelList[i] = paste(head(rownames(wordsDistr[order(wordsDistr[[i]], decreasing = T),]), n = label.len), collapse = delim)
  }
  
  return(labelList)
}

