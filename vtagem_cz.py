
import sys

from misc_func import *
from misc_fun_em import *
from misc_func_cz import *
from vterbi_tagger import *
from forward_backward import *

max_epochs = 10

tr_file, tst_file, raw_file = parse_args_em()

tr_data = load_data_cz(tr_file)

sys.exit(-1)
tst_data = load_data(tst_file)

raw_data = load_raw_data(raw_file)

#Compute and store all transmission counts in memory
tags = compute_tr_counts(tr_data)
#print("#\n#Tags:",tags)

#Compute and store all emission counts in memory
tag_dict = compute_em_counts(tr_data)
#print("#\n#Tag Dictionary:",tag_dict)

for epoch in range(max_epochs):
    
    #Step 1: Viterbi on test data
    #Get the best path from viterbi decoder
    best_path_tags = decoder(tst_data, tag_dict, tags)
    #print("#\n#Best path tags",best_path_tags)
    
    #Compute all accuracies, return format: 0.89 (for 89%)
    accuracy, known_accuracy, novel_accuracy = compute_accuracy(best_path_tags,tst_data)
    
    #Print the final output for autograder
    print("Tagging accuracy (Viterbi decoding):\t%2.2f%%\t(known:\t%2.2f%% novel:\t%2.2f%%)"%(100*accuracy, 100*known_accuracy,100*novel_accuracy))
    
    #Step 2: Forward Backward on raw data
    #Get the posterior tags from forward backward decoder
    posterior_tags, perplexity = FBdecoder(raw_data, tag_dict, tags)
    #print("#\n#Posterior tags from forward backward",posterior_tags)
    
    ############# Output for autograder #############
    print("Iteration %d:\tModel perplexity per untagged raw word:\t%4.4f"%(epoch,perplexity))
    
    ####### TODO: Recompute transition and emission counts ########
    ########################################################################
    retrain_seq = reformat_train(raw_data,posterior_tags,tr_data)
    reinitialise_counts()
    
    #Compute and store all transmission counts in memory
    tags = compute_tr_counts(retrain_seq)
    #print("#\n#Tags:",tags)
    
    #Compute and store all emission counts in memory
    tag_dict = compute_em_counts(retrain_seq)
#print("#\n#Tag Dictionary:",tag_dict)

#Final decoding after 10 EM iterations
#Get the posterior tags from forward backward decoder
posterior_tags, _ = FBdecoder(tst_data, tag_dict, tags)
#print("#\n#Posterior tags from forward backward",posterior_tags)

test_output_file = 'test-output'
write_to_test_output(test_output_file,tst_data,posterior_tags)
