import sys

#Get the relevant info from the user
def parse_args_em():
    if len(sys.argv) < 4:
        raise ValueError("Correct usage: python3 vtagem train.file test.file raw.file")

    return sys.argv[1],sys.argv[2],sys.argv[3]

#Read the files into some sequences (list of lists, a temporary variable f_data, we will not need these once we calcuate the counts later)
def load_raw_data(file):
    with open(file) as f:
        data = f.read().strip().split('\n')
        
        f_data = []
        seq = None
        for line in data:
            if "###" in line:
                #BOS/EOS
                if seq:
                    seq.append(['###',None])
                    #Add sequence to f_data - list of all sequences
                    f_data.extend(seq)
                    seq = []
                else:
                    seq = [['###',None]]
            else:
                seq.append([line, None])
    return f_data

def reformat_train(raw_data, raw_tags, tr_seq):
    f_data = [[word[0], tag] for word, tag in zip(raw_data, raw_tags)]
    f_data.extend(tr_seq)
    return f_data
