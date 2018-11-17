from misc_func import *
from datetime import datetime

#Implementation of viterbi tagger as per the handout
def decoder(seq, tag_dict, tags_with_bos):
    tags = tags_with_bos.copy()
    mu, backpointer, true_score = defaultdict(list), defaultdict(list), defaultdict() #true_score contains cross entropies for all sentences
    n = len(seq)    #seq is the test word sequence, format - "long string", check H.2 in Homework
    true_mu = 0
    
    ############# Initialisations #############
    for t in tags:
        mu[t] = [-float("inf")]*(n)
        backpointer[t] = [0]*(n)

    mu['###'] = [0] + [-float("inf")]*(n-1)
    backpointer['###'] = [0]*(n)

    ############# Viterbi Decoder #############
    for i,[cur_word,true_tag] in zip(range(1,n), seq[1:n]):
        prev_word = seq[i-1][0]
        #Get a list of tags for the cur_word (if it is a novel word or if prev word was novel, then use all tags)
        tags_of_cur_word = tags if cur_word not in tag_dict else tag_dict[cur_word][0]
        tags_of_prev_word = tags if prev_word not in tag_dict else tag_dict[prev_word][0]
        for tag in tags_of_cur_word:
            for prev_tag in tags_of_prev_word:
                #Log probability
                #lp = get_tr_prob(tag,prev_tag) + get_em_prob(cur_word,tag)
                #Alternate probabilities using 1-count smoothing, Use either of the three lps
                lp = one_count_prob_tt(tag,prev_tag) + one_count_prob_tw(cur_word,tag)
                #Alternate probabilities using ADDL BACKOFF smoothing, Use either of the three lps
                #lp = get_tr_prob_ALBOFF(tag,prev_tag) + get_em_prob_ALBOFF(cur_word,tag)
                #If this is the first word of new sentence, restart the mu from 1 (log prob to 0)
                
                cur_mu = lp if prev_word == '###' else mu[prev_tag][i-1]+lp
                if cur_mu > mu[tag][i]:
                    mu[tag][i] = cur_mu
                    backpointer[tag][i] = prev_tag
                
                ##true prev tag and cur tag then save mu for computing perplexity
                if tag == true_tag and prev_tag == seq[i-1][1]:
                    true_mu += lp
                    if cur_word == '###':
                        true_score[i] = true_mu
                        true_mu = 0

    ############# Backtrack #############
    #Collect the backpointers into tags, n = 35 for ictest including BOS and EOS
    tag_track = [0]*(n-1) + ['###']
    #print("#tag_track[%d]:"%(n-1),tag_track[n-1],"mu[%d]:"%(n-1),mu[tag_track[n-1]][n-1])
    for i in reversed(range(1,n)):
        tag_track[i-1] = backpointer[tag_track[i]][i] if backpointer[tag_track[i]] else None
        #Uncomment the following line to view the intermediate values of mu, best probability path till i_th state
        #print("#tag_track[%d]:"%(i-1),tag_track[i-1],"mu[%d]:"%(i-1),mu[tag_track[i-1]][i-1])

    ############# Output for autograder #############
    print("Model perplexity per tagged test word:\t%4.3f"%(math.exp(-sum(true_score.values())/(n-1))))
    return tag_track
