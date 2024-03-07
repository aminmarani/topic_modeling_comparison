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
#' @param nits number of iterations for LDA gibbs sampling
#' @param burnin for LDA gibbs sampling
#' @param alpha prevalence hyperparameter in collabsed gibbs sampling for LDA
#' @param eta sets the topic-word hyperparameter for collapsed gibbs sampling in LDA
#' @param rp.s a param between 0-1 controlling the sparsity of random projection for Spectral initialization
#' @param rp.p dimensionality of the random projections for Spectral initialization
#' @param tSNE_init.dims if you use K=0 for spectral initialization, the algorithm uses tSNE for starting and only uses tSNE_init.dims (i.e., default = 50) to initialize
#' @param no_below removes the terms that only appear in no_below docs (default = 5)
#' @param no_above remove the terms that appear in more than no_above proportion of the docs (default = 0.5)
#' @param save_flage if you want to save the model (default = False)
#' @param save_path the path to store the model if save_flag = T (default = 'default' but it will change to specificiation of the model if you keep it as default; e.g., max_itr)
run_stm <- function(docs,topic_n=10,verbose=F,reportevery=5,prevalence='',content='', model_type='LDA',max_itr=500,emtol = 1e-05,LDAbeta=T,interactions=T,ngroups=1, sigma.prior = 0, gamma.prior='Pooled',kappa.prior='L1',nits=50,burnin=25, alpha=-1, eta= 0.1,rp.s = 0.05, rp.p=3000, tSNE_init.dims = 50,no_below=5,no_above=0.5,save_flag = F,save_path='default')
{
  #if no value is provide, we set it to the default STM uses, 50/K
  if (alpha == -1)
    alpha = 50/topic_n
  
	###checking libraries
	if (! require('reader'))
		stop('reader package is not installed! You need to isntall this package to continue.')	
	if (! require('tm'))
    {
		# stop('tm package is not installed! You need to isntall this package to continue.')	
        print('installing TM...')
        install.packages('tm')
    }
	if (! require('stm'))
    {
        # stop('stm package is not installed! You need to isntall this package to continue.')	
        print('installing STM...')
        install.packages('stm')
            
    }


	###loading libraries
	library(readr)
	library(stm)

	###adding sources
	# source('./coherence.R')

	###adjusting data
	# text <- docs$text

	# data <- data.frame(as.character(1:length(text)), stringsAsFactors = FALSE)
	# names(data) <- "id"
	# data$text <- text

	processed <- textProcessor(docs$text, metadata = docs,stem = FALSE,onlycharacter = F, removestopwords = F,  removenumbers = F,  removepunctuation = F)
	out <- prepDocuments(processed$documents, processed$vocab, processed$meta, lower.thresh=no_below, upper.thresh=as.integer(length(docs$text)*no_above))
	docs <- out$documents
	vocab <- out$vocab
	meta <-out$meta

    
	#running STM
	if (nchar(prevalence)>0 && nchar(content)>0)
    {
	  STM <- stm(documents = out$documents, vocab = out$vocab,
	                K = topic_n, prevalence =~ as.factor(out$meta$prevalence)+as.factor(out$meta$content),
                    content=~content,data = out$meta,
	                 max.em.its = max_itr,
	                emtol = emtol,LDAbeta=F, interactions=interactions,
	                ngroups = ngroups, gamma.prior = gamma.prior, 
	                kappa.prior = kappa.prior, control= list(nits=nits,
	                burnin=burnin, alpha= alpha, eta= eta, rp.s = rp.s,
	                rp.p=rp.p, tSNE_init.dims = tSNE_init.dims),
	                init.type = "LDA", verbose = verbose,reportevery=reportevery,seed = 12345)#data = out$meta,
    }
	else if (nchar(prevalence)>0)
    {
        STM <- stm(documents = out$documents, vocab = out$vocab,
	                K = topic_n, prevalence =~ as.factor(out$meta$prevalence)+as.factor(out$meta$content) ,#out$meta$prevalence
	                data = out$meta, max.em.its = max_itr,
                  emtol = emtol,LDAbeta=T, interactions=interactions,
                  ngroups = ngroups,gamma.prior = gamma.prior, 
	                kappa.prior = kappa.prior, control= list(nits=nits,
	                burnin=burnin, alpha= alpha, eta= eta, rp.s = rp.s,
	                rp.p=rp.p, tSNE_init.dims = tSNE_init.dims),
	                init.type = "LDA", verbose = verbose,reportevery=reportevery,seed = 12345)
    }
	else if (nchar(content)>0)
    {
      # You can't call stm without prevalence and only content...
      # You can either call stm with prevalence or with both content and prevalnece
	  STM <- stm(documents = out$documents, vocab = out$vocab,
	                K = topic_n,content=~as.factor(out$meta$content),
	                max.em.its = max_itr, data = out$meta, 
	                emtol = emtol,LDAbeta=F, interactions=interactions,
	                ngroups = ngroups,gamma.prior = gamma.prior, 
	                kappa.prior = kappa.prior, control= list(nits=nits,
	                burnin=burnin, alpha= alpha, eta= eta, rp.s = rp.s,
	                rp.p=rp.p, tSNE_init.dims = tSNE_init.dims),
	                init.type = "LDA", verbose = verbose,reportevery=reportevery,seed = 12345)
    }
	else
	  STM <- stm(documents = out$documents, vocab = out$vocab,
	                K = topic_n, max.em.its = max_itr, data = out$meta, 
	                emtol = emtol, LDAbeta=F, interactions=interactions, 
	                ngroups = ngroups, gamma.prior = gamma.prior, 
	                kappa.prior = kappa.prior, control= list(nits=nits,
	                burnin=burnin, alpha= alpha, eta= eta, rp.s = rp.s,
	                rp.p=rp.p, tSNE_init.dims = tSNE_init.dims),
	                init.type = "LDA", verbose = verbose,reportevery=reportevery,seed = 12345)
  
    #storing the results
    if (save_flag)
      {
        if (save_path == 'default')#change the path
            save_path = paste('K',topic_n,'epochs',max_itr,'model_type',model_type,
                              Sys.Date(),Sys.time(),sep='_')
        save(STM, ascii=FALSE, file=paste('STM_',save_path))
      }
    top_terms <- getLabels(STM,topic_n)
	return(list(STM,top_terms,meta))
}



#' extracting top terms
#'
#' @description recieves an STM model and returns top-N terms
#' @param STM, the trained model
#' @param topic_n number of topics (the same as the model was trained on)
#' @param label.len number of top terms for each topic (default = 50)
getLabels <- function(STM,topic_n,label.len = 50){
      #storing top terms
      top.terms = matrix(nrow = topic_n,ncol = label.len)
      log.beta = STM$beta$logbeta[[1]]
      for (i in 1:topic_n)
      {
        sx = sort(log.beta[i,],index.return=T,decreasing = T)
        top.terms[i,] =  STM$vocab[sx$ix[1:label.len]]
      }
    
      return(top.terms)
    }
