from misc_func import *

tr_file, tst_file = parse_args()

tr_data = load_data(tr_file)

tst_data = load_data(tst_file)

#Compute and store all transmission counts in memory
tags = compute_tr_counts(tr_data)
#print("#\n#Tags:",tags)

#Compute and store all emission counts in memory
tag_dict = compute_em_counts(tr_data)
#print("#\n#Tag Dictionary:",tag_dict)

my_tags = tags
my_words = list(tag_dict.keys()).copy()
my_words.append('OOV')

### For IC2 dataset
#my_tags = ['H','C','###']
#my_words = ['1','2','3','3a','3b','3c','###']

transition_prob = {}

#Check transition probabilities
for tagp in my_tags:
    sum = 0
    for tagc in my_tags:
        transition_prob[(tagc,tagp)] = math.exp(get_tr_prob(tagc,tagp))
        sum += transition_prob[(tagc,tagp)]
    print("tags on",(tagp),"sum to:",sum)

#print("Transition probabilities:", transition_prob)


emission_prob = {}
#Check transition probabilities
for tag in my_tags:
    sum = 0
    for word in my_words:
        emission_prob[(word,tag)] = math.exp(get_em_prob(word,tag))
        sum += emission_prob[(word,tag)]
    print("words on",(tag),"sum to:",sum)
#print("Emission probabilities:", emission_prob)


################
one_count_check = True
if one_count_check == True:
    print("\n\nOne Count smoothing probabilities:")

    transition_prob = {}
    #Check transition probabilities
    for tagp in my_tags:
        sum = 0
        for tagc in my_tags:
            transition_prob[(tagc,tagp)] = math.exp(one_count_prob_tt(tagc,tagp))
            sum += transition_prob[(tagc,tagp)]
        print("1C tags on",(tagp),"sum to:",sum)
    #print("1C Transition probabilities:", transition_prob)


    emission_prob = {}
    #Check transition probabilities
    for tag in my_tags:
        sum = 0
        for word in my_words:
            emission_prob[(word,tag)] = math.exp(one_count_prob_tw(word,tag))
            sum += emission_prob[(word,tag)]
        print("1C words on",(tag),"sum to:",sum)
#print("1C Emission probabilities:", emission_prob)
