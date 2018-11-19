import sys
import math
from collections import defaultdict
import codecs
#Global
orig_count_tt = defaultdict()
current_count_tt, tag_dict, tags, singleton_tt, singleton_tw = defaultdict(), defaultdict(list), [], [], []
one_count_singleton_tt, one_count_singleton_em = defaultdict(),defaultdict()
lamda = 1
curr_n_tokens = 0
n_BOS = 0
orig_n_tokens = 0
#Get the relevant info from the user
def parse_args():
    if len(sys.argv) < 3:
        raise ValueError("Correct usage: python3 vtag train.file test.file")
    
    return sys.argv[1],sys.argv[2]

#Read the files into some sequences (list of lists, a temporary variable f_data, we will not need these once we calcuate the counts later)
def load_data(file):

    f = open(file, 'r')
    f_data = []
    seq = None
    # with open(file, 'r') as f:
    for line in f:
        # raw = f.read()
        # raw = raw.decode("utf-8")
        # data = f.read().strip().split('\n')
        
        
        # for line in data:
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
            # print line
            line_split = line.strip().split('/')

            seq.append([line_split[0], line_split[1]])
    return f_data

def get_tr_str(tagc,tagp):
    return tagc+'/'+tagp
#Computes transition counts and stores in current_count_tt
#Also computes tags counts and also stores in current_count_tt
def compute_tr_counts(data):
    #Start counting
    prev_tag = None
    global singleton_tt
    global tags
    for i in range(len(data)):
        cur_tag = data[i][1]
        #Update the tag count for current tag and track number of unique tags seen in training which will be used for smoothing
        tags.append(cur_tag)
        current_count_tt[cur_tag] = 1 if cur_tag not in current_count_tt else current_count_tt[cur_tag] + 1
        #Update the pair counts
        if prev_tag is not None:
            current_count_tt[get_tr_str(cur_tag,prev_tag)] = 1 if get_tr_str(cur_tag,prev_tag) not in current_count_tt else current_count_tt[get_tr_str(cur_tag,prev_tag)] + 1
            #Singleton counts for 1-count smoothing
            if current_count_tt[get_tr_str(cur_tag,prev_tag)] == 1:
                singleton_tt.append((cur_tag,prev_tag))
            elif (cur_tag,prev_tag) in singleton_tt:
                singleton_tt.remove((cur_tag,prev_tag))

        #Update the prev tag
        prev_tag = cur_tag
    
    #Create a list of unique tags seen in training
    tags = set(tags)
    singleton_tt = set(singleton_tt)
    #Extra BOS at the end
    current_count_tt['###'] -= 1
    #Compute counts for singleton
    create_dict_of_singleton_counts_tt()
    return tags

#Computes emission counts and stores in the current_count_tt
#Also creates tag_dict {word -> [[tag1, tag2, tag3 ...], count_of_word]}
def compute_em_counts(data,raw_data=None):
    global curr_n_tokens
    global n_BOS
    global orig_count_tt
    global singleton_tw
    global orig_n_tokens
    for i in range(len(data)):
        [cur_word, cur_tag] = data[i]
        curr_n_tokens += 1
        n_BOS += 1 if cur_word == '###' else 0
        #Collect all the observed words in tag_dict with their counts }
        if cur_word not in tag_dict:
            tag_dict[cur_word] = [[cur_tag],0]
        elif cur_tag not in tag_dict[cur_word][0]:
            tag_dict[cur_word][0].append(cur_tag)
        tag_dict[cur_word][1] += 1
        #Counts word tag pairs
        current_count_tt[(cur_word,cur_tag)] = 1 if (cur_word,cur_tag) not in current_count_tt else current_count_tt[(cur_word,cur_tag)] + 1
        #Singleton counts for 1-count smoothing
        if current_count_tt[(cur_word,cur_tag)] == 1:
            singleton_tw.append((cur_word,cur_tag))
        elif current_count_tt[(cur_word,cur_tag)] == 2:
            singleton_tw.remove((cur_word,cur_tag))

    if raw_data is not None:
        for i in range(len(raw_data)):
            [cur_word,_] = raw_data[i]
            if cur_word not in tag_dict:
                tag_dict[cur_word] = [[None],0]
    #Extra BOS at the end of the string
    current_count_tt[('###','###')] -= 1
    tag_dict['###'][1] -= 1
    curr_n_tokens -= 1
    n_BOS -= 1
    singleton_tw = set(singleton_tw)
    #print("#\n#Count Dictionary current_count_tt:",current_count_tt,tags)
    #print("#\n#Tag_dict:",tag_dict)
    #Compute counts for one count singleton
    create_dict_of_singleton_counts_em()
    orig_count_tt = current_count_tt.copy()
    orig_n_tokens = curr_n_tokens
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
    
    acc = float(sum(match))/len(match) if len(match) != 0 else 0
    novel_acc = float(sum(novel))/len(novel) if len(novel) != 0 else 0
    known_acc = float(sum(match)-sum(novel))/(len(match)-len(novel)) if (len(match)-len(novel)) != 0 else 0
    return acc, known_acc, novel_acc

#Add lambda transition probabilities
def get_tr_prob(tagc, tagp):

    use_count = current_count_tt
    if get_tr_str(tagc,tagp) not in use_count:
        return math.log(float(lamda) / (use_count[tagp] + lamda*(len(tags)))) if lamda > 0 else -float("inf") # -1 in the len(tags) is for ###
    else:
        return math.log((use_count[get_tr_str(tagc,tagp)] + lamda) / (use_count[tagp] + lamda*(len(tags))))

#Add lambda emission probabilities
def get_em_prob(word, tag):
 
    use_count = current_count_tt
    #Special handling of BOS, EOS as mentioned in H.6.1 of the homework
    if word == '###':
        if tag == '###':
            return 0
        else:
            return -float("inf")
    elif tag == '###':
        return -float("inf")
    #If its a novel word
    if (word, tag) not in use_count:
        #Length of tag_dict = number of words in the vocab  minus the ### and plus the novel OOV
        return math.log(float(lamda) / (use_count[tag] + lamda*(len(tag_dict))))  if lamda > 0 else -float("inf")
    elif word not in tag_dict:
        return math.log((use_count[('OOV',tag)] + lamda) / (use_count[tag] + lamda*(len(tag_dict))))
    #General case of known words
    else:
        return math.log((use_count[(word,tag)] + lamda) / (use_count[tag] + lamda*(len(tag_dict))))


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


def recompute_count(new_count_tt, new_n_tokens):
    global current_count_tt
    global orig_n_tokens
    global curr_n_tokens
    for item, cnt in new_count_tt.items():
        if item in orig_count_tt:
            current_count_tt[item] = orig_count_tt[item] + cnt
        else:
            current_count_tt[item] = cnt
    curr_n_tokens = orig_n_tokens + new_n_tokens

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
    global curr_n_tokens
    pb = float(current_count_tt[cur_tag])/curr_n_tokens
    # print("backoff prob tra, cur_tag:",cur_tag,pb,"curr_n_tokens:",curr_n_tokens)
    return pb

def backoff_tw(word):
    global curr_n_tokens
    global n_BOS
    if word in tag_dict:
        pb = float(tag_dict[word][1] + 1) / (curr_n_tokens + len(tag_dict) - n_BOS)
    else:
        pb = 1.0 / (curr_n_tokens + len(tag_dict) - n_BOS)
    #print("backoff prob emi, word:",word,pb,"curr_n_tokens:",curr_n_tokens,"V:",len(tag_dict))
    return pb

def one_count_prob_tt(cur_tag, prev_tag):
    one_count_lamda = 1 + sing_tt(prev_tag)
    
    if get_tr_str(cur_tag, prev_tag) in current_count_tt:
        pr =  float(current_count_tt[get_tr_str(cur_tag, prev_tag)] + one_count_lamda*backoff_tt(cur_tag)) / \
            (current_count_tt[prev_tag] + one_count_lamda)
    else:
        pr =  float(one_count_lamda*backoff_tt(cur_tag)) / \
            (current_count_tt[prev_tag] + one_count_lamda)

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

    if (word, tag) in current_count_tt:
        pr =  float(current_count_tt[(word, tag)] + one_count_lamda*backoff_tw(word)) / \
            (current_count_tt[tag] + one_count_lamda)
    else:
        pr =  float(one_count_lamda*backoff_tw(word)) / \
        (current_count_tt[tag] + one_count_lamda)

    return math.log(pr)

def get_prunable_tags(thresh):
    prunable_tags = []
    remove_word_tag = {}
    for t in tags:
        if current_count_tt[t] < thresh:
            prunable_tags.append(t)
    #print("Original tags list:",tags,"of length:",len(tags))
    #print("Number of singletone tags:",len(prunable_tags),":",prunable_tags)
    for word,val in tag_dict.items():
        for t in prunable_tags:
            
            if t in val[0] and len(val[0]) == 1:
                prunable_tags.remove(t)
                remove_word_tag.pop(word,None)
            elif t in val[0] and len(val[0])>1:
                if word not in remove_word_tag:
                    remove_word_tag[word] = t
                else:
                    prunable_tags.remove(t)
                    remove_word_tag.pop(t,None)
            elif t not in val[0]:
                remove_word_tag.pop(word,None)

    for t in prunable_tags:
        for word, val in tag_dict.items():
            if t in val[0]:
                tag_dict[word][0].remove(t)

    return prunable_tags, tag_dict
