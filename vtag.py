
import sys

from misc_func import *
from vterbi_tagger import *
from forward_backward import *
from check_prob import *
tr_file, tst_file = parse_args()

tr_data = load_data(tr_file)

tst_data = load_data(tst_file)

#Compute and store all transmission counts in memory
tags = compute_tr_counts(tr_data)
#print("#\n#Tags:",tags)

#Compute and store all emission counts in memory
tag_dict = compute_em_counts(tr_data)
#print("#\n#Tag Dictionary:",tag_dict)

#Check for probabilities (conditional probabilities sum to 1 property)
#check_probability(tags,tag_dict)

#Get the best path from viterbi decoder
best_path_tags = decoder(tst_data, tag_dict, tags)
#print("#\n#Best path tags",best_path_tags)

#Compute all accuracies, return format: 0.89 (for 89%)
accuracy, known_accuracy, novel_accuracy = compute_accuracy(best_path_tags,tst_data)

#Print the final output for autograder
print("Tagging accuracy (Viterbi decoding):\t%2.2f%% (known:\t%2.2f%% novel:\t%2.2f%%)"%(100*accuracy, 100*known_accuracy,100*novel_accuracy))

#Get the posterior tags from forward backward decoder
posterior_tags, _, _ = FBdecoder(tst_data, tag_dict, tags)
#print("#\n#Posterior tags from forward backward",posterior_tags)

#Compute all accuracies, return format: 0.89 (for 89%)
ps_accuracy, ps_known_accuracy, ps_novel_accuracy = compute_accuracy(posterior_tags,tst_data)

#Print the final output for autograder
print("Tagging accuracy (posterior decoding):\t%2.2f%% (known:\t%2.2f%% novel:\t%2.2f%%)"%(100*ps_accuracy, 100*ps_known_accuracy,100*ps_novel_accuracy))

test_output_file = 'test-output'
write_to_test_output(test_output_file,tst_data,posterior_tags)
