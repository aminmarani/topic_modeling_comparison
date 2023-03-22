# for now, always use top-n for generating labels from which to calculate coherence.
##################################if linux please change ../ to ./
source("./labellers/topn_topic_labeler.R")

writeLabels <- function(model = public.model, label_len = 25, outdir = "topic_interpretability/data", outfile = "topics.txt") {
  # get labels
  long.labels <- getLabels(model = model, label_len = 25)
  # write to specified file
  dir.create(outdir, showWarnings = FALSE)
  write(long.labels, paste(c(outdir, "/", outfile), collapse = ""))
}

writeDocuments <- function(docs, outdir = "topic_interpretability/ref_corpus/", corpus.name = "my_corpus") {
  # get documents (assumes that docs contains a field names text)
  doc.texts <- gsub("\n", " ", docs$text)
  segmented.texts <- split(doc.texts, (seq(length(doc.texts))-1) %/% 50)
  
  # if the directory doesn't exist, create it
  corpus.dir <- paste(outdir, corpus.name, sep="")
  dir.create(corpus.dir, showWarnings = FALSE)
  
  # write the documents out to files
  corpus.path <- paste(c(outdir, corpus.name, "/corpus"), collapse="")
  sapply(seq(length(segmented.texts)), writeIt, segmented.texts, corpus.path)
}

# helper function for writing subsets of documents to files
writeIt <- function(index, seg.texts, file.prefix) {
  foo <- paste(file.prefix, index-1, sep=".")
  write(unlist(seg.texts[index]), foo)
}

# run the external coherence calculation script
calculateCoherence <- function(datadir = "topic_interpretability/data/", datafile = "topics.txt", ref.corpus.dir = "topic_interpretability/ref_corpus/wiki", metric = "npmi", label.cardinality = "-t 5 10 15 20") {
  # run Lau et al.'s python scripts via command line
  # NB: some of these parameters could potentially be tweaked in their formatting to be more generic/adaptable
  wc.file <- paste("topic_interpretability/wordcount/wc-", datafile, sep="")
  wc.command <- sprintf("python topic_interpretability/ComputeWordCount.py %s%s %s > %s", datadir, datafile, ref.corpus.dir, wc.file)
  system(wc.command)
  oc.file <- paste("topic_interpretability/results/oc-", datafile, sep="")
  oc.command <- sprintf("python topic_interpretability/ComputeObservedCoherence.py %s%s %s %s %s > %s", datadir, datafile, metric, wc.file, label.cardinality, oc.file)
  system(oc.command)
  
  # read the coherence results into a dataframe
  cohereLines <- readLines(oc.file)
  cohereLines <- head(cohereLines, -3) # remove the summary stats
  cohereScores <- lapply(cohereLines, function(x)
      {substr(x, 2, regexpr("]", x)[1] - 1)})
  topicLabels <- lapply(cohereLines, function(x)
      {substr(x, regexpr(")", x)[1] + 2, nchar(x))})
  topicCoherence <- cbind(cohereScores, topicLabels)
  return(topicCoherence)
}

coherenceSweep <- function(documents, n.topics, stopwords="freqPOS.stop", corpus.name = "my_corpus", ref.corpus.path = "topic_interpretability/ref_corpus/")
{
  # NB: assumes that documents have already been written into a reference corpus directory in the standard location with the name of the corpus (i.e., "<ref.corpus.path>/<corpus.name>")
  
  for (n in n.topics)
  {
    # build the model with this number of topics
    createModel(documents, stopwords=stopwords, label_method="topn", label_len=6,
                model_use_public=F, n.topics=n)
    # make the appropriate file names for this number of topics
    outfile <- paste("topics-", corpus.name, n, ".txt", sep="")
    ref.corpus.dir <- paste(ref.corpus.path, corpus.name, sep="")
    # write out the labels and calculate coherence
    writeLabels(outfile = outfile)
    calculateCoherence(datafile = outfile, ref.corpus.dir = ref.corpus.dir)
  }
}
