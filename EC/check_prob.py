from misc_func import *

def check_probability(my_tags,tag_dict):

    my_words = list(tag_dict.keys()).copy()
    my_words.append('OOV')

    transition_prob = {}

    #Check transition probabilities
    for tagp in my_tags:
        sum = 0
        for tagc in my_tags:
            transition_prob[(tagc,tagp)] = math.exp(get_tr_prob(tagc,tagp))
            sum += transition_prob[(tagc,tagp)]
            #print("#T:",(tagc,tagp),transition_prob[(tagc,tagp)])
        print("tags on",(tagp),"sum to:",sum)

    emission_prob = {}
    #Check transition probabilities
    for tag in my_tags:
        sum = 0
        for word in my_words:
            emission_prob[(word,tag)] = math.exp(get_em_prob(word,tag))
            sum += emission_prob[(word,tag)]
            #print("#E:",(word,tag),emission_prob[(word,tag)])
        print("words on",(tag),"sum to:",sum)
    #print("Emission probabilities:", emission_prob)


    transition_prob = {}
    
    #Check transition probabilities
    for tagp in my_tags:
        sum = 0
        for tagc in my_tags:
            transition_prob[(tagc,tagp)] = math.exp(one_count_prob_tt(tagc,tagp))
            sum += transition_prob[(tagc,tagp)]
        #print("#T:",(tagc,tagp),transition_prob[(tagc,tagp)])
        print("tags on",(tagp),"sum to:",sum)

    emission_prob = {}
    #Check transition probabilities
    for tag in my_tags:
        sum = 0
        for word in my_words:
            emission_prob[(word,tag)] = math.exp(one_count_prob_tw(word,tag))
            sum += emission_prob[(word,tag)]
        #print("#E:",(word,tag),emission_prob[(word,tag)])
        print("words on",(tag),"sum to:",sum)
#print("Emission probabilities:", emission_prob)
