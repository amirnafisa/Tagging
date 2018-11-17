Hidden Markov Model for tagging sentences.

Implementations using Viterbi and Forward Backward Algorithm

Usage:
python3 vtag.py train_file test_file

Another application is semi-supervised tagging using EM algorithm

Usage:
python3 vtagem.py train_file test_file raw_file

The syntax of the train_file and test_file is <word>/<tag>\n. 
Beginning and end of sentence is given by ###/###.

The syntax of raw_file is <word>\n
Beginning and end of sentence is given by ###.
