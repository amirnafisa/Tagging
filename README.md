***Hidden Markov Model for tagging sentences and semi-supervised learning using Viterbi and Forward Backward Algorithm***

For supervised learning,
Usage:
*python3 vtag.py train_file test_file*

For semi-supervised learning,
Usage:
*python3 vtagem.py train_file test_file raw_file*

The syntax of the train_file and test_file is \<word\>/\<tag\>. 
Beginning and end of sentence is given by ###/###.

The syntax of raw_file is \<word\>
Beginning and end of sentence is given by ###.
