#checking installed package to warn users
# if (! require('rJava'))
# 	stop('rJava package is not installed! You need to isntall this package to continue. \n **You need Java to install this package**')

# if (! require('mallet'))
# 	stop('mallet package is not installed! You need to isntall this package to continue.')

# if (! require('reshape2'))
# 	stop('reshape2 package is not installed! You need to isntall this package to continue.')	

# if (! require('reticulate'))
# 	stop('reticulate package is not installed! You need to isntall this package to continue.')	

# if (! require('qdapTools'))
# 	stop('qdapTools package is not installed! You need to isntall this package to continue.')	

if (! require('reader'))
	stop('reader package is not installed! You need to isntall this package to continue.')	

# if (! require('dplyr'))
# 	stop('dplyr package is not installed! You need to isntall this package to continue.')	

# if (! require('ggplot2'))
# 	stop('ggplot2 package is not installed! You need to isntall this package to continue.')	

# if (! require('jsonlite'))
# 	stop('jsonlite package is not installed! You need to isntall this package to continue.')	


# test <- function(inp)
# {
# 	print(inp)
# 	return (13123.1232)
# }


# #loading libraries
# library(rJava)
# library(mallet)
library(readr)
# library(dplyr)
# .jinit(parameters="-Xmx12g") #Give rJava enough memory
# library(ggplot2)
# library(reshape2)
# library(jsonlite)
# library(reticulate)
# library(qdapTools)


#'running STM for one-single run 
#' @param docs documents with text and other meta-data
#' @param topic_n number of topics
#' @param verbose showing the process including topic labels every N iterations
#' @param prevalence Right-side of a formula showing the prevalence relation. It should be vars in the dataframe. e.g., days+party
#' @param content Right-side of a formlua for content. It should be vars of the dataframe you pass as docs
#' @param model_type initialization type including LDA, Random, Spectral. If you are using Spectral, set K=0 so the algorithm find the best K for you!
#' @param max_itr maximum iteration STM part takes. It does not include LDA or Spectral iterations.
#' @param emtol the tolerance to finish the iterative algorithm before reaching to max_itr
#' @param LDAbeta When there is no content variable set to TRUE. If set to False the model perform SAGE style topic updates
#' @param interactions includes interaction between content variable and latent topics. Set to FALSE to reduce the model to no interactions
#' @param ngroups A number equal or bigger than 1. This divides the corpus into multiple memory division and makes the algorithm converge faster. It could end up with differnt results with different ngroups
#' @param sigma.prior A scalar between 0-1. Sets the strength of regularization of covariance matrix. If topics are highly correlated, setting this value >0 would be helpful.
#' @param gamma.prior Prior estimation method for prevalence. Values=c('Pooled','L1')
#' @param kappa.prior prior estimation method for content. Values=c('L1','Jeffreys')
#' @param number of iterations for LDA gibbs sampling
#' @param buring for LDA gibbs sampling
#' @param alpha prevalence hyperparameter in collabsed gibbs sampling for LDA
#' @param eta sets the topic-word hyperparameter for collapsed gibbs sampling in LDA
#' @param rp.s a param between 0-1 controlling the sparsity of random projection for Spectral initialization
#' @param rp.p dimensionality of the random projections for Spectral initialization
#' @param tSNE_init.dims if you use K=0 for spectral initialization, the algorithm uses tSNE for starting and only uses tSNE_init.dims (i.e., default = 50) to initialize
#'
run_stm <- function(docs,topic_n=10,verbose=T,prevalence='',content='', model_type='LDA',max_itr=500,emtol = 1e-05,LDAbeta=T,interactions=T,ngroups=1, sigma.prior = 0, gamma.prior='Pooled',kappa.prior='L1',nits=50,burnin=25, alpha=-1, eta= 0.1,rp.s = 0.05, rp.p=3000, tSNE_init.dims = 50,no_below=5,no_above=0.5)
{
  #if no value is provide, we set it to the default STM uses, 50/K
  if (alpha == -1)
    alpha = 50/topic_n
  
	###checking libraries
	if (! require('reader'))
		stop('reader package is not installed! You need to isntall this package to continue.')	
	if (! require('stm'))
		stop('stm package is not installed! You need to isntall this package to continue.')	
	if (! require('tm'))
		stop('tm package is not installed! You need to isntall this package to continue.')	


	###loading libraries
	library(readr)
	library(stm)

	###adding sources
	source('./coherence.R')

	###adjusting data
	# text <- docs$text

	# data <- data.frame(as.character(1:length(text)), stringsAsFactors = FALSE)
	# names(data) <- "id"
	# data$text <- text

	processed <- textProcessor(docs$text, metadata = docs,stem = FALSE,onlycharacter = T)
	out <- prepDocuments(processed$documents, processed$vocab, processed$meta,, lower.thresh=no_below, upper.thresh=as.integer(length(docs$text)*no_above))
	docs <- out$documents
	vocab <- out$vocab
	meta <-out$meta

	#running STM
	if (nchar(prevalence)>0 && nchar(content)>0)
	  STM <- stm(documents = out$documents, vocab = out$vocab,
	                K = topic_n, prevalence =~ prevalence ,content=content,
	                data = out$meta, max.em.its = max_itr,
	                emtol = emtol,LDAbeta=FALSE, interactions=interactions,
	                ngroups = ngroups, gamma.prior = gamma.prior, 
	                kappa.prior = kappa.prior, control= list(nits=nits,
	                burnin=burnin, alpha= alpha, eta= eta, rp.s = rp.s,
	                rp.p=rp.p, tSNE_init.dims = tSNE_init.dims),
	                init.type = "LDA", verbose = FALSE,seed = 12345)
	else if (nchar(prevalence)>0)
    STM <- stm(documents = out$documents, vocab = out$vocab,
	                K = topic_n, prevalence =~ prevalence ,
	                max.em.its = 75, data = out$meta, max.em.its = max_itr,
                  emtol = emtol,LDAbeta=T, interactions=interactions,
                  ngroups = ngroups,gamma.prior = gamma.prior, 
	                kappa.prior = kappa.prior, control= list(nits=nits,
	                burnin=burnin, alpha= alpha, eta= eta, rp.s = rp.s,
	                rp.p=rp.p, tSNE_init.dims = tSNE_init.dims),
	                init.type = "LDA", verbose = FALSE,seed = 12345)
	else if (nchar(content)>0)
	  STM <- stm(documents = out$documents, vocab = out$vocab,
	                K = topic_n,content=content,
	                max.em.its = max_itr, data = out$meta, 
	                emtol = emtol,LDAbeta=F, interactions=interactions,
	                ngroups = ngroups,gamma.prior = gamma.prior, 
	                kappa.prior = kappa.prior, control= list(nits=nits,
	                burnin=burnin, alpha= alpha, eta= eta, rp.s = rp.s,
	                rp.p=rp.p, tSNE_init.dims = tSNE_init.dims),
	                init.type = "LDA", verbose = FALSE,seed = 12345)
	else
	  STM <- stm(documents = out$documents, vocab = out$vocab,
	                K = topic_n, max.em.its = max_itr, data = out$meta, 
	                emtol = emtol, LDAbeta=F, interactions=interactions, 
	                ngroups = ngroups, gamma.prior = gamma.prior, 
	                kappa.prior = kappa.prior, control= list(nits=nits,
	                burnin=burnin, alpha= alpha, eta= eta, rp.s = rp.s,
	                rp.p=rp.p, tSNE_init.dims = tSNE_init.dims),
	                init.type = "LDA", verbose = FALSE,seed = 12345)
  #storing top terms
	top.terms = matrix(nrow = topic_n,ncol = 50)
  log.beta = STM$beta$logbeta[[1]]
  for (i in 1:topic_n)
  {
    sx = sort(log.beta[i,],index.return=T,decreasing = T)
    top.terms[i,] =  STM$vocab[sx$ix[1:50]]
  }
	return(list(STM,top.terms))
}


# ####defining functions####

# #' Running mallet instance and returning the model
# #'
# #' @param docs documents to run LDA on. docs should have three variables/fields. 1. text, including texts of the documetns 2. title, an id ora title for the doc and 3.?
# #' @param n.topics number of topics
# #' @param stop_words a list of stop words
# #' @param word_min_freq any term with occurences in less or equal to $word_min_freq$ doc, will be removed from vocabulary. Defaults to 2
# #' @param rand.seed a seed for random initialization. Defaukts to 54321
# findTopics <- function(docs, n.topics, stop_words, word_min_freq=2,rand.seed=54321L){
#   #Use mallet as in sample code
#   print("Building mallet instance ...")
#   # replace single smart quote with single straight quote, so as to catch stopword contractions
#   docs$text <- gsub("[\u2018\u2019]", "'", docs$text)
#   mallet.instances <- mallet.import(docs$title, docs$text, stop_words, token.regexp = "\\b[\\d\\p{L}][\\d\\p{L}\\\\_\\/&'-]+\\p{L}\\b")
#   #Unused regex: (?u)\b[\w][\w\-\.]+[\w]\b  #\\p{L}[\\p{L}\\p{P}]+\\p{L}  (?u)\\b[\\w\\p{L}][\\w\\p{L}-.]+[\\w\\p{L}]\\b
  
#   ## Create a topic trainer object.
#   print("Building topic trainer ...")
#   topic.model <<- MalletLDA(num.topics=n.topics)
#   topic.model$model$setRandomSeed(rand.seed)
  
#   ## Load our documents. We could also pass in the filename of a 
#   ##  saved instance list file that we build from the command-line tools.
#   print("Loading documents in mallet instance into topic trainer ...")
#   topic.model$loadDocuments(mallet.instances)
  
#   ## Optimize hyperparameters every 20 iterations, 
#   ##  after 10 burn-in iterations.
#   topic.model$setAlphaOptimization(20, 10)
  
#   ## Now train a model.
#   ##  We can specify the number of iterations. Here we'll use a large-ish round number.
#   print("Training model ...")
#   topic.model$train(1000)#1000
#   topic.model$maximize(50)#50
#   print("Training complete.")
  
#   doc.topics <- data.frame(mallet.doc.topics(topic.model, smoothed=T, normalized=T)) #docs * topics matrix (topic makeup of documents)
#   topic.words.count <- mallet.topic.words(topic.model, smoothed=T, normalized=F) #topics * words (word distribution over topics)
  
#   word.freq <- mallet.word.freqs(topic.model)
#   word.freq$term.freq <- colSums(mallet.topic.words(topic.model, normalized=F), na.rm=TRUE) #fixes issue with term count but not doc count
#   #gm
#   word.freq.filt <- word.freq
#   word.freq.filt$id <- 1:nrow(word.freq.filt)
#   word.freq.filt <- filter(word.freq.filt, term.freq >= word_min_freq) #Filters out low frequency words
#   word.freq <- word.freq[word.freq.filt$id,] #Removes word from word.freq
  
#   topic.words.count <- topic.words.count[,word.freq.filt$id] #Removes columns of words filtered out
#   topic.words <- sweep(topic.words.count, 1, rowSums(topic.words.count), `/`) #Normalizes topic distribution
  
#   wordlist <- as.character(word.freq.filt[,1]) #List of words in document. Order matchs order of words in topic.words
  
#   model = list(doc_distr = doc.topics, word_distr = topic.words, words = wordlist, 
#             malletModel = topic.model, wordFreq = word.freq, word_distr_count = topic.words.count) #malletModel given for easy implementation of topn labeler 
#                                                                                                    #but it is only saved for the session
#   return(model)
# }


# #' Get top-n (label_len) terms as labels of topics
# #'
# #' @param docs documents the model was trained on
# #' @param model the trained LDA model
# #' @param stop_words list of stop words
# #' @param label_len number of top terms to appear as labels
# #' @param delim delimiter to sepearate top terms
# getLabels <- function(docs, model, stop_words, label_len, delim=" "){
#   wordsDistr = data.frame(t(model$word_distr))
#   rownames(wordsDistr) = model$words
  
#   labelList = rep(NA, length(model$doc_distr))
#   for (i in 1:length(model$doc_distr)) {
#     labelList[i] = paste(head(rownames(wordsDistr[order(wordsDistr[[i]], decreasing = T),]), n = label_len), collapse = delim)
#   }
  
#   return(labelList)
# }


# writeDocuments <- function(docs, outdir = "", corpus.name = "my_corpus") {
#   # get documents (assumes that docs contains a field names text)
#   doc.texts <- gsub("\n", " ", docs$text)
#   segmented.texts <- split(doc.texts, (seq(length(doc.texts))-1) %/% 50)
  
#   # if the directory doesn't exist, create it
#   corpus.dir <- paste(outdir, corpus.name, sep="")
#   dir.create(corpus.dir, showWarnings = FALSE)
  
#   # write the documents out to files
#   corpus.path <- paste(c(outdir, corpus.name, "/corpus"), collapse="")
#   sapply(seq(length(segmented.texts)), writeIt, segmented.texts, corpus.path)
# }

# # helper function for writing subsets of documents to files
# writeIt <- function(index, seg.texts, file.prefix) {
#   foo <- paste(file.prefix, index-1, sep=".")
#   write(unlist(seg.texts[index]), foo)
# }

# # run the external coherence calculation script
# calculateCoherence <- function(datadir = "topic_interpretability/data/", datafile = "topics.txt", ref.corpus.dir = "topic_interpretability/ref_corpus/wiki", metric = "npmi", label.cardinality = "-t 5 10 15 20") {
#   # run Lau et al.'s python scripts via command line
#   # NB: some of these parameters could potentially be tweaked in their formatting to be more generic/adaptable
#   wc.file <- paste("wc-", datafile, sep="")
#   #download python files
#   wc.command <- sprintf("python ComputeWordCount.py %s%s %s > %s", datadir, datafile, ref.corpus.dir, wc.file)
#   system(wc.command)
#   oc.file <- paste("oc-", datafile, sep="")
#   oc.command <- sprintf("python ComputeObservedCoherence.py %s%s %s %s %s > %s", datadir, datafile, metric, wc.file, label.cardinality, oc.file)
#   system(oc.command)
  
#   # read the coherence results into a dataframe
#   cohereLines <- readLines(oc.file)
#   cohereLines <- head(cohereLines, -3) # remove the summary stats
#   cohereScores <- lapply(cohereLines, function(x)
#       {substr(x, 2, regexpr("]", x)[1] - 1)})
#   topicLabels <- lapply(cohereLines, function(x)
#       {substr(x, regexpr(")", x)[1] + 2, nchar(x))})
#   topicCoherence <- cbind(cohereScores, topicLabels)
#   return(topicCoherence)
# }

# coherenceSweep <- function(documents, n.topics, stopwords="freqPOS.stop", corpus.name = "my_corpus", ref.corpus.path = "topic_interpretability/ref_corpus/")
# {
#   # NB: assumes that documents have already been written into a reference corpus directory in the standard location with the name of the corpus (i.e., "<ref.corpus.path>/<corpus.name>")
  
#   for (n in n.topics)
#   {
#     # build the model with this number of topics
#     createModel(documents, stopwords=stopwords, label_method="topn", label_len=6,
#                 model_use_public=F, n.topics=n)
#     # make the appropriate file names for this number of topics
#     outfile <- paste("topics-", corpus.name, n, ".txt", sep="")
#     ref.corpus.dir <- paste(ref.corpus.path, corpus.name, sep="")
#     # write out the labels and calculate coherence
#     writeLabels(outfile = outfile)
#     calculateCoherence(datafile = outfile, ref.corpus.dir = ref.corpus.dir)
#   }
# }