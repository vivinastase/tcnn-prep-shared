import os

import constants
import numpy as np
import pickle as pickle


class Index(object):
    def __init__(self):
        self._ent_index = dict()



    def ent_to_ind(self,ent):
        if ent not in self._ent_index:
            self._ent_index[ent] = len(self._ent_index.keys())
        return self._ent_index[ent]

    def load_index(self,dir_name):
        if os.path.exists(os.path.join(dir_name,constants.entity_ind)):
            self._ent_index = pickle.load(open(os.path.join(dir_name,constants.entity_ind),'rb'))

        else:
            print("Index not found, creating one.")

    def save_index(self,dir_name):
        pickle.dump(self._ent_index,open(os.path.join(dir_name,constants.entity_ind),'wb'))

    def ent_vocab_size(self):
        return len(self._ent_index)


class Path(object):
    def __init__(self, s, t, preps):
        assert isinstance(s, int) and isinstance(t, int)
        assert isinstance(preps, np.ndarray)
        self.s = s # source
        self.t = t # target
        self.preps = preps
        self.pairs = [s,t]

    def __repr__(self):
        rep = "{} {}".format(self.s,self.t)
        return rep

    def __eq__(self, other):
        if not isinstance(other,Path):
            return False
        equal = self.s == other.s and self.t == other.t
        return equal

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        hash_p = self.s.__hash__() + self.t.__hash__()
        return hash_p

    def source(self):
        return self.s
    
    def target(self):
        return self.t

def read_dataset(path, results_dir,dev_mode=True,max_examples = float('inf'),gen_neg=True):

    index = Index()
    index.load_index(results_dir)
    data_set = {}
    (data_set['train'],data_set['dev'],data_set['test']) = read_file(os.path.join(path,'all_data'),index,max_examples)

    constants.dev_samples = len(data_set['dev'])/3   ## this is how many instances will be sampled from dev for evaluation during training
    data_set['ent_vocab'] = index.ent_vocab_size()
    constants.vocab_size = index.ent_vocab_size()
    index.save_index(results_dir)
    return data_set

def read_file(f_name, index, max_examples):
    data = []
    count = 0
    index_test = set()
    print("Reading data from file " + f_name)
    with open(f_name) as f:
        for line in f:
            if count >= max_examples:
                return data
            parts = line.strip().split("\t")
            
            if count == 0:
                constants.output_dims = len(parts)-1
                print("Output length: {}".format(constants.output_dims))

            s,t = parts[0].split(" ")
            labels = parts[1:]
            assert len(labels)==constants.output_dims
            labels = np.asarray([float(x) for x in labels],dtype='float32')
            p = Path(index.ent_to_ind(s), index.ent_to_ind(t),labels)
            data.append(p)
            count += 1
            
            index_test.add(s)
            index_test.add(t)
            
    print("Unique words: {} / {}".format(len(index_test),index.ent_vocab_size()))
    print("Pairs: {}".format(len(data)))
    ## now split into train, dev, test
    print("Splitting data ... ")
    train = []
    index_train = []
    test = []
    dev = []
    for pair in data:
        s = pair.source()
        t = pair.target()
        print("Pair: (",s,",",t,")")
        if (pair.source() in index_train) and (pair.target() in index_train) and len(test)+len(dev) < 15000 :
            if (len(dev) < 5000):
                dev.append(pair)
                print("\tto dev")
            elif (len(test) < 10000):
                test.append(pair)
                print("\tto test")
            else:
                train.append(pair)
                index_train.append(pair.source())
                index_train.append(pair.target())
        else:
            train.append(pair)
            index_train.append(pair.source())
            index_train.append(pair.target())

    print("Train: {} instances".format(len(train)))
    print("Dev: {} instances".format(len(dev)))
    print("Test: {} instances".format(len(test)))
    return (train, dev, test)

