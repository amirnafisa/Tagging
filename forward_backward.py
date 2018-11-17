from misc_func import *
from datetime import datetime
#Implementation of viterbi tagger as per the handout
def FBdecoder(seq, tag_dict, tags):
    #We need BOS in out tags list for forward backward iterations
    #tags = tags_without_bos.copy()
    #tags.add('###')
    
    alpha, beta, backpointer, score, cross_entropy = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict() #cross_entropy contain cross entropies for all sentences then compute cross_entropy using these
    n = len(seq)    #seq is the test word sequence, format - "long string", check H.2 in Homework

    ############# Initialisations #############
    for t in tags:
        alpha[t] = [-float("inf")]*(n)
        beta[t] = [-float("inf")]*(n)
        score[t] = [-float("inf")]*(n)
    
    tag_track, max_score = ['###'] + [0]*(n-1), [-float("inf")]*(n)
    #score['###'] = [-float("inf")]*(n)

    ############# Computing alpha #############
    alpha['###'] = [0] + [-float("inf")]*(n)
    alpha['###'][0] = 0
    for i,[cur_word,_] in zip(range(1,n), seq[1:n]):
        prev_word = seq[i-1][0]
        #Get a list of tags for the cur_word (if it is a novel word or if prev word was novel, then use all tags)
        tags_of_cur_word = tags if cur_word not in tag_dict else tag_dict[cur_word][0]
        tags_of_prev_word = tags if prev_word not in tag_dict else tag_dict[prev_word][0]
        
        for tag in tags_of_cur_word:
            for prev_tag in tags_of_prev_word:
                #Log probability
                #lp = get_tr_prob(tag,prev_tag) + get_em_prob(cur_word,tag)
                #Alternate probability smoothed by one-count smoothing method, Use only one of the two lp equations
                lp = one_count_prob_tt(tag,prev_tag) + one_count_prob_tw(cur_word,tag)
                #Alternate probability smoothed by ADDL BACKOFF method, Use only one of the three lp equations
                #lp = get_tr_prob_ALBOFF(tag,prev_tag) + get_em_prob_ALBOFF(cur_word,tag)
                
                #If this is the first word of new sentence, restart the mu from 1 (log prob to 0)
                alpha[tag][i] = lp if prev_word == '###' else logsumexp(alpha[tag][i], alpha[prev_tag][i-1] + lp)


    Z = alpha['###'][n-1]

    ############# Computing beta #############
    beta['###'][n-1] = 0
    for i,[cur_word,_] in zip(reversed(range(1,n)), reversed(seq[1:n])):
        prev_word = seq[i-1][0]

        #Get a list of tags for the cur_word (if it is a novel word or if prev word was novel, then use all tags)
        tags_of_cur_word = tags if cur_word not in tag_dict else tag_dict[cur_word][0]
        tags_of_prev_word = tags if prev_word not in tag_dict else tag_dict[prev_word][0]
    
        if cur_word == '###':
            cross_entropy[i] = cross_entropy_sentence if i != n-1 else 0
            cross_entropy_sentence = 1
            next_idx = i
    
        for tag in tags_of_cur_word:
            ###########Compute p(Ti = ti | w vector) = alpha[ti][i] * beta[ti][i] / Z
            score[tag][i] = alpha[tag][i] + beta[tag][i] - Z        #Insert code at line 13 -> Implemented
            ####################################################
            if i == next_idx: #Need only compute for any one of the word, alpha(t)*beta(t) + alpha(t2)*beta(t2) +...
                cross_entropy_sentence = logsumexp(cross_entropy_sentence, alpha[tag][i] + beta[tag][i]) if cross_entropy_sentence <= 0 else alpha[tag][i] + beta[tag][i]
            
            for prev_tag in tags_of_prev_word:
                #Log probability
                #lp = get_tr_prob(tag,prev_tag) + get_em_prob(cur_word,tag)
                #Alternate probability smoothed by one-count smoothing method, Use only one of the three lp equations
                lp = one_count_prob_tt(tag,prev_tag) + one_count_prob_tw(cur_word,tag)
                #Alternate probability smoothed by ADDL BACKOFF method, Use only one of the three lp equations
                #lp = get_tr_prob_ALBOFF(tag,prev_tag) + get_em_prob_ALBOFF(cur_word,tag)
                #If this is the first word of new sentence, restart the mu from 1 (log prob to 0)
                beta[prev_tag][i-1] = lp if prev_word == '###' else logsumexp(beta[prev_tag][i-1], beta[tag][i] + lp)
                ##########Compute p(T(i-1) = t(i-1), Ti = ti | w vector) = alpha[t(i-1)][i-1] * p * beta[ti][i]/Z and sum them to the total score for ti
                #score[tag][i] += alpha[prev_tag][i-1] + lp + beta[tag][i]

            if score[tag][i] > max_score[i]:
                max_score[i] = score[tag][i]
                tag_track[i] = tag
    
    cross_entropy[i] = cross_entropy_sentence

    #Uncomment below lines to print all alpha and beta values. You can directly cross check with spreadsheet for ic dataset (no smoothing)
    #print("Forward Backward Tags:", tag_track)
    #for i in range(1,n-1):
    #    print("i:",i,"alpha[C]:%.2e"%math.exp(alpha['C'][i]),"alpha[H]:%.2e"%math.exp(alpha['H'][i]),"beta[C]:%.2e"%math.exp(beta['C'][i]),"beta[H]:%.2e"%math.exp(beta['H'][i]),"alpha[C]beta[C]:%.2e"%math.exp(alpha['C'][i]+beta['C'][i]),"alpha[H]beta[H]:%.2e"%math.exp(alpha['H'][i]+beta['H'][i]))
    
    return tag_track, (math.exp(-sum(cross_entropy.values())/(n-1)))
