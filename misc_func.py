import sys
import math
from collections import defaultdict
#Global
count_tt, tag_dict, tags, singleton_tt, singleton_tw = defaultdict(), defaultdict(list), [], [], []
one_count_singleton_tt, one_count_singleton_em = defaultdict(),defaultdict()
lamda = 1
n_tokens = 0
n_BOS = 0
#Get the relevant info from the user
def parse_args():
    if len(sys.argv) < 3:
        raise ValueError("Correct usage: python3 vtag train.file test.file")
    
    print("#Note: Add Lambda Smoothing, lambda = ",lamda)
    return sys.argv[1],sys.argv[2]

#Read the files into some sequences (list of lists, a temporary variable f_data, we will not need these once we calcuate the counts later)
def load_data(file):
    with open(file) as f:
        data = f.read().strip().split('\n')
        
        f_data = []
        seq = None
        for line in data:
            if "###/###" in line:
                #BOS/EOS
                if seq:
                    seq.append(['###','###'])
                    #Add sequence to f_data - list of all sequences
                    f_data.extend(seq)
                    seq = []
                else:
                    seq = [['###','###']]
            else:
                line = line.split('/')
                seq.append([line[0], line[1]])
    return f_data

def get_tr_str(tagc,tagp):
    return tagc+'/'+tagp
#Computes transition counts and stores in count_tt
#Also computes tags counts and also stores in count_tt
def compute_tr_counts(data):
    #Start counting
    prev_tag = None
    
    for i in range(len(data)):
        global tags
        global singleton_tt
        
        cur_tag = data[i][1]
        #Update the tag count for current tag and track number of unique tags seen in training which will be used for smoothing
        tags.append(cur_tag)
        count_tt[cur_tag] = 1 if cur_tag not in count_tt else count_tt[cur_tag] + 1
        #Update the pair counts
        if prev_tag is not None:
            count_tt[get_tr_str(cur_tag,prev_tag)] = 1 if get_tr_str(cur_tag,prev_tag) not in count_tt else count_tt[get_tr_str(cur_tag,prev_tag)] + 1
            #Singleton counts for 1-count smoothing
            if count_tt[get_tr_str(cur_tag,prev_tag)] == 1:
                singleton_tt.append((cur_tag,prev_tag))
            elif (cur_tag,prev_tag) in singleton_tt:
                singleton_tt.remove((cur_tag,prev_tag))

        #Update the prev tag
        prev_tag = cur_tag
    
    #Create a list of unique tags seen in training
    tags = set(tags)
#tags.remove('###')
    singleton_tt = set(singleton_tt)
    #Extra BOS at the end
    count_tt['###'] -= 1
    #Compute counts for singleton
    create_dict_of_singleton_counts_tt()
    return tags

#Computes emission counts and stores in the count_tt
#Also creates tag_dict {word -> [[tag1, tag2, tag3 ...], count_of_word]}
def compute_em_counts(data):
    global n_tokens
    global n_BOS
    global singleton_tw
    singleton_tw = list(singleton_tw)
    for i in range(len(data)):
        [cur_word, cur_tag] = data[i]
        n_tokens += 1
        n_BOS += 1 if cur_word == '###' else 0
        #Collect all the observed words in tag_dict with their counts }
        if cur_word not in tag_dict:
            tag_dict[cur_word] = [[cur_tag],0]
        elif cur_tag not in tag_dict[cur_word][0]:
            tag_dict[cur_word][0].append(cur_tag)
        tag_dict[cur_word][1] += 1
        #Counts word tag pairs
        count_tt[(cur_word,cur_tag)] = 1 if (cur_word,cur_tag) not in count_tt else count_tt[(cur_word,cur_tag)] + 1
        #Singleton counts for 1-count smoothing 
        if count_tt[(cur_word,cur_tag)] == 1:
            singleton_tw.append((cur_word,cur_tag))
        elif count_tt[(cur_word,cur_tag)] == 2:
            singleton_tw.remove((cur_word,cur_tag))
    #Extra BOS at the end of the string
    count_tt[('###','###')] -= 1
    tag_dict['###'][1] -= 1
    n_tokens -= 1
    n_BOS -= 1
    singleton_tw = set(singleton_tw)
    #print("#\n#Count Dictionary count_tt:",count_tt,tags)
    #print("#\n#Tag_dict:",tag_dict)
    #Compute counts for one count singleton
    create_dict_of_singleton_counts_em()
    return tag_dict

#Computes accuracy given two lists (also computes accuracy of novel words and known words in the lists)
def compute_accuracy(output, target_seq):
    
    if len(output) != len(target_seq):
        raise ValueError("Length of output and target do not match when computing accuracy.")
    
    match, novel = [], []
    for i in range(len(output)):
        
        if output[i] != '###':
            match.append((output[i] == target_seq[i][1])*1)
            novel.append((output[i] == target_seq[i][1])*1) if target_seq[i][0] not in tag_dict else None

    if len(match) == 0:
        return 0, 0, 0
    
    acc = sum(match)/len(match) if len(match) != 0 else 0
    novel_acc = sum(novel)/len(novel) if len(novel) != 0 else 0
    known_acc = (sum(match)-sum(novel))/(len(match)-len(novel)) if (len(match)-len(novel)) != 0 else 0
    return acc, known_acc, novel_acc

#Add lambda transition probabilities
def get_tr_prob(tagc, tagp):
    if get_tr_str(tagc,tagp) not in count_tt:
        return math.log(lamda / (count_tt[tagp] + lamda*(len(tags)))) if lamda > 0 else -float("inf") # -1 in the len(tags) is for ###
    else:
        return math.log((count_tt[get_tr_str(tagc,tagp)] + lamda) / (count_tt[tagp] + lamda*(len(tags))))

#Add lambda emission probabilities
def get_em_prob(word, tag):
    #Special handling of BOS, EOS as mentioned in H.6.1 of the homework
    if word == '###':
        if tag == '###':
            return 0
        else:
            return -float("inf")
    elif tag == '###':
        return -float("inf")
    #If its a novel word
    if (word, tag) not in count_tt:
        #Length of tag_dict = number of words in the vocab  minus the ### and plus the novel OOV
        pr = math.log(lamda / (count_tt[tag] + lamda*(len(tag_dict))))  if lamda > 0 else -float("inf")
    #General case of known words
    else:
        pr = math.log((count_tt[(word,tag)] + lamda) / (count_tt[tag] + lamda*(len(tag_dict))))
    return pr

#Computes the sum of log probabilities -> check section H.4
def logsumexp(lp, lq):
    if lp == -float("inf"):
        return lq
    if lq == -float("inf"):
        return lp
    
    return max(lp,lq) + math.log(1 + math.exp(-(abs(lp - lq))))

#Write tagged test sequence to file in current directory
def write_to_test_output(file,sequence,output_tags):
    with open(file,'w') as f:
        for word_seq, tag in zip(sequence,output_tags):
            f.write(word_seq[0]+'/'+tag+'\n')


########  1-count smoothing ###########
def create_dict_of_singleton_counts_tt():
    for prev_tag in tags:
        tag_pairs = []
        for cur_tag in tags:
            tag_pairs.append((cur_tag,prev_tag))
        one_count_singleton_tt[prev_tag] = len(singleton_tt.intersection(set(tag_pairs)))

def create_dict_of_singleton_counts_em():
    for tag in tags:
        word_tag_pairs = []
        for word, _ in tag_dict.items():
            word_tag_pairs.append((word,tag))
        one_count_singleton_em[tag] = len(singleton_tw.intersection(set(word_tag_pairs)))


def sing_tt(prev_tag):
    return one_count_singleton_tt[prev_tag]
    #global singleton_tt
    #tag_pairs = []
    #for tag in tags:
    #    tag_pairs.append((tag,prev_tag))
    #count = len(singleton_tt.intersection(set(tag_pairs)))
    #return count

def sing_tw(tag):
    return one_count_singleton_em[tag]
    #global singleton_tw
    #word_tag_pairs = []
    #for word, tags in tag_dict.items():
    #    word_tag_pairs.append((word,tag))
    #count = len(singleton_tw.intersection(set(word_tag_pairs)))
    #return count

def backoff_tt(cur_tag):
    global n_tokens
    pb = count_tt[cur_tag]/n_tokens
    #print("backoff prob tra, cur_tag:",cur_tag,pb,"n_tokens:",n_tokens)
    return pb

def backoff_tw(word):
    global n_tokens
    global n_BOS
    if word in tag_dict:
        pb = (tag_dict[word][1] + 1) / (n_tokens + len(tag_dict) - n_BOS)
    else:
        pb = 1 / (n_tokens + len(tag_dict) - n_BOS)
    #print("backoff prob emi, word:",word,pb,"n_tokens:",n_tokens,"V:",len(tag_dict))
    return pb

def one_count_prob_tt(cur_tag, prev_tag):
    one_count_lamda = 1 + sing_tt(prev_tag)
    if get_tr_str(cur_tag, prev_tag) in count_tt:
        pr =  (count_tt[get_tr_str(cur_tag, prev_tag)] + one_count_lamda*backoff_tt(cur_tag)) / \
            (count_tt[prev_tag] + one_count_lamda)
    else:
        pr =  (one_count_lamda*backoff_tt(cur_tag)) / \
            (count_tt[prev_tag] + one_count_lamda)

    #print("tran: (cur_tag,prev_tag):",(cur_tag, prev_tag),pr)
    return math.log(pr)
def one_count_prob_tw(word, tag):
    if word != '###' and tag == '###':
        return -float("inf")
    if word == '###' and tag == '###':
        return 0
    if word == '###' and tag != '###':
        return -float("inf")

    one_count_lamda = 1 + sing_tw(tag)

    if (word, tag) in count_tt:
        pr =  (count_tt[(word, tag)] + one_count_lamda*backoff_tw(word)) / \
            (count_tt[tag] + one_count_lamda)
    else:
        pr =  (one_count_lamda*backoff_tw(word)) / \
        (count_tt[tag] + one_count_lamda)

    return math.log(pr)

#### Reinitialise counts before recounting for M step of EM
def reinitialise_counts():
    global tags
    global singleton_tt
    global singleton_tw
    global n_tokens
    global n_BOS
    count_tt.clear()
    tag_dict.clear()
    del tags
    del singleton_tt
    del singleton_tw
    tags = []
    singleton_tt = []
    singleton_tw = []
    n_tokens = 0
    n_BOS = 0


################ Add lambda with backoff smoothing #################3
##Add lambda transition backoff probability
def get_backoff_tr(tagc):
    return count_tt[tagc] / n_tokens

###emission backoff probability for add lambda
def get_backoff_em(word):
    if word not in tag_dict:
        return lamda / (n_tokens - n_BOS + lamda*len(tag_dict))
    else:
        return (tag_dict[word][1] + lamda) / (n_tokens - n_BOS + lamda*len(tag_dict))

#Add lambda transition probabilities
def get_tr_prob_ALBOFF(tagc, tagp):
    #Backoff Prob
    Pbo = get_backoff_tr(tagc)
    if get_tr_str(tagc,tagp) not in count_tt:
        return math.log(lamda*len(tags)*Pbo / (count_tt[tagp] + lamda*(len(tags)))) if lamda > 0 else -float("inf") # -1 in the len(tags) is for ###
    else:
        return math.log((count_tt[get_tr_str(tagc,tagp)] + lamda*len(tags)*Pbo) / (count_tt[tagp] + lamda*(len(tags))))

#Add lambda emission probabilities
def get_em_prob_ALBOFF(word, tag):
    #Special handling of BOS, EOS as mentioned in H.6.1 of the homework
    if word == '###':
        if tag == '###':
            return 0
        else:
            return -float("inf")
    ### Back off probabilities
    Pbo = get_backoff_em(word)
    #If its a novel word
    if (word, tag) not in count_tt:
        #Length of tag_dict = number of words in the vocab  minus the ### and plus the novel OOV
        return math.log(lamda*len(tag_dict)*Pbo / (count_tt[tag] + lamda*(len(tag_dict))))  if lamda > 0 else -float("inf")
    #General case of known words
    else:
        return math.log((count_tt[(word,tag)] + lamda*len(tag_dict)*Pbo) / (count_tt[tag] + lamda*(len(tag_dict))))

