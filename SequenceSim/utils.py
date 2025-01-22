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




def markov_simulations(seq_parameters, n):
    """Repeat simulations n times""" 
    seqs = [markov_seq_string(init_freq = seq_parameters['init_freq'], \
                               transition_dic = seq_parameters['transition_dic'], \
                               transition_states = seq_parameters['transition_states'], \
                               n = seq_parameters['n'], \
                               order = seq_parameters['order']) for i in range(n)]
    
    seqs_encoded = numpy.array([one_hot_encode(seq_i, seq_parameters['transition_states']) for seq_i in seqs])
    return seqs_encoded

def markov_simulations_motif(seq_parameters, n, file, index):
    """Repeat simulations n times""" 
    seqs = [markov_seq_string(init_freq = seq_parameters['init_freq'], \
                               transition_dic = seq_parameters['transition_dic'], \
                               transition_states = seq_parameters['transition_states'], \
                               n = seq_parameters['n'], \
                               order = seq_parameters['order']) for i in range(n)]
    
    if file is not None:
        mots = [motifer(file) for i in range(n)]
        start = numpy.random.randint(low=index[0], high=index[1], size = n)
        for i, s in enumerate(seqs):
            seqs[i] = s[:start[i]] + mots[i] + s[start[i]+len(mots[i]):]

    seqs_encoded = numpy.array([one_hot_encode(seq_i, seq_parameters['transition_states']) for seq_i in seqs])
    return seqs_encoded


def experiments_markov_binary(cases_seq_parameters, controls_seq_parameters, \
                              cases_n, controls_n, \
                              cases_motif_file = None, cases_index = None, \
                              controls_motif_file = None, controls_index = None):
    """Allow different simulations for cases and controls allowing different parameters for cases and controls"""
    X_cases = markov_simulations_motif(cases_seq_parameters, cases_n, cases_motif_file, cases_index)
    y_cases = numpy.repeat(1, cases_n)
    
    X_controls = markov_simulations_motif(controls_seq_parameters, controls_n, controls_motif_file, controls_index)
    y_controls = numpy.repeat(0, controls_n).reshape(controls_n, 1)
    
    X = numpy.vstack([X_cases, X_controls])
    y = numpy.append(y_cases, y_controls).reshape(controls_n + cases_n, 1)
    
    return X, y