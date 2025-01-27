import numpy
def markov_seq_string(init_freq, transition_dic, transition_states, n, order):
    """"Simulate a sequence using markov process"""
    seq = numpy.random.choice(a = [*init_freq], p = [*init_freq.values()])
    n_new = n - 1 - (order - 1)
    for i in range(n_new):
        prior = ''.join(seq[-order:])
        prob = transition_dic[prior]
        post = numpy.random.choice(a= transition_states, p = prob)
        seq += post
    return seq

def one_hot_encode(seq, transition_states):
    """One hot endcode all characters in the string"""
    mapping = dict(zip(transition_states, range(len(transition_states))))    
    seq2 = [mapping[i] for i in seq]
    return numpy.eye(len(transition_states))[seq2]

def motifer(mfile):
    with open(mfile,  'r', encoding='utf-8') as file:
        next(file)
        rfile = [line.strip('\n') for line in file]
    pfm = []
    for l in rfile:
        #pfm.append([float(i) for i in l.split()])
        pfm.append([float(i) for i in l.split()])
    pfm = list(map(numpy.array, zip(*pfm)))
    pwm = [i/sum(i) for i in pfm]
    tfbs = ""
    for i in pwm:
        tfbs += (numpy.random.choice(a= ["A", "C", "G", "T"], p = i))
    return tfbs


def markov_simulations_motif(seq_parameters, n, files, indexes, presence):
    """Repeat simulations n times""" 
    seqs = numpy.array([markov_seq_string(init_freq = seq_parameters['init_freq'], \
                               transition_dic = seq_parameters['transition_dic'], \
                               transition_states = seq_parameters['transition_states'], \
                               n = seq_parameters['n'], \
                               order = seq_parameters['order']) for i in range(n)])
    # add motifs
    if files is not None:
        for file, index, p in zip(files, indexes, presence):
            if file is not None:
                
                m = numpy.random.choice(range(n), size= round(n*p), replace=False)
                
                mots = [motifer(file) for i in m]
                start = numpy.random.randint(low=index[0], high=index[1], size = len(m))
                
                for t, i in enumerate(m):
                    seqs[i] = seqs[i][:start[t]] + mots[t] + seqs[i][start[t]+len(mots[t]):]
                

    seqs_encoded = numpy.array([one_hot_encode(seq_i, seq_parameters['transition_states']) for seq_i in seqs])
    return seqs_encoded


def experiments_markov_binary(cases_seq_parameters, controls_seq_parameters, \
                              cases_n, controls_n, \
                              cases_motif_file = None, cases_index = None, cases_presence = None, \
                              controls_motif_file = None, controls_index = None, controls_presence = None):
    """Allow different simulations for cases and controls allowing different parameters for cases and controls"""
    X_cases = markov_simulations_motif(cases_seq_parameters, cases_n, \
                                       cases_motif_file, cases_index, cases_presence)
    y_cases = numpy.repeat(1, cases_n)
    
    X_controls = markov_simulations_motif(controls_seq_parameters, controls_n, \
                                          controls_motif_file, controls_index, controls_presence)
    y_controls = numpy.repeat(0, controls_n).reshape(controls_n, 1)
    
    X = numpy.vstack([X_cases, X_controls])
    y = numpy.append(y_cases, y_controls).reshape(controls_n + cases_n, 1)
    
    return X, y