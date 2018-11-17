import codecs

#Read the files into some sequences (list of lists, a temporary variable f_data, we will not need these once we calcuate the counts later)
def load_data_cz(file):
    with open(file, encoding='utf8') as f:
        raw = f.read().decode('utf8')
        sys.exit(-1)
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
    sys.exit(-1)
    return f_data
