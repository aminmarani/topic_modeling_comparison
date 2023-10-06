import numpy as np

def picking_topic(dist,method='dynamic',min_p = 0.5,top_p=0.75):
    '''
    Receives topic distributions that sums up to 1.0 and return the cut for top topic based on the selected method
    @param dist: distribution list (default: numpy)
    @param method: 'dynamic' the largest difference or 'top_p' all the top topics within top_p percent
    @param min_p: the minimum distribution if dynamic is selected
    @param top_p: top topic percentage if 'top_p' selected as the method

    @returns: an int number for the topic to cut
    '''

    #sending erros if the input is not prepared
    if method not in ['dynamic','top_p']:
        raise RuntimeError("the method for similarity has to be dynamic or top_p")
    if len(dist) == 0:
        raise RuntimeError("The topic distribution vector is empty")

    #indices of sort values
    idx = np.argsort(dist)[::-1]
    
    if method == 'dynamic': #selecting the biggest difference
        diffs = [dist[idx[i]] - dist[idx[i+1]] for i in range(len(idx)-1)]
        #finding the biggest difference
        max_ind = np.argmax(diffs)
        #saving topics that are higher than max_ind topic distribution (note that diffs in 1 size smaller than number of topics)
        cut_topic_ind = idx[max_ind]#what is the index of cut-off topic
        cut_topic_dist = dist[cut_topic_ind] #what is the distribution of cut-off topic
        topics = [1 if ds >= cut_topic_dist else 0 for ds in dist]

        #check if dynamic method selected at least min_p distribution
        if np.sum([dist[i] for i in range(len(topics)) if topics[i]==1]) <= min_p:
            #if so let's run top_p with top_p = min_p
            method = 'top_p'
            top_p = min_p

    if method == 'top_p':
        topics = [0]*len(dist)
        cum_sum = 0.0
        for id in idx:
            if cum_sum <= top_p:#if the sorted distribution is within the range, add it to the list
                topics[id] = 1
                cum_sum += dist[id]
            else:
                break

    return topics



# def topic_pair_sim(topics1,topics2,sim_fun):
#     '''
#     compute similarity for all pairs from topics1 and topics2 regarding similarity function

#     @param topics1: a list of lists containing top terms for each topic
#     @param topics2: a list of lists containing top terms for each topic
#     @param sim_fun: a similarity function for two sets

#     @returns: returns a matrix of similarity between all pairs of topics1 and topics2
#     '''
#     sim_score = np.zeros((len(topics1),len(topics2)))

#     for i in range(len(topics1)):
#         for j in range(len(topics2)):
#             sim_score[i,j] = sim_fun(topics[i,:],topics[j,:])

#     return sim_score


def doc_sim():
    #union(all_top_topics for doc1) intersect with union(all_top_topics for doc2) ==> how about adding a memory so we don't need to redo these computations everytime!
    return 0

    