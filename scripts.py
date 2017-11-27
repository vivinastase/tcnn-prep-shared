import os

import numpy as np


def vocab_size():
    base = 'paraphrases/'
    files = {'test','train','test'}
    words = set()
    for file in files:
        with open(os.path.join(base,file)) as f:
            for line in f:
                parts = line.strip().split('\t')
                s,t = parts[0].split(' ')
                words.add(s)
                words.add(t)
    print("Vocab Size {}".format(len(words)))

def create_splits():
    base = 'paraphrases/'
    prob = 0.0625
    dev_writer = open(os.path.join(base,'dev'),'w')
    trn_writer = open(os.path.join(base,'trn'),'w')
    with open(os.path.join(base,'train')) as f:
        for line in f:
            if np.random.uniform(size=1) <= 0.0625:
                dev_writer.write(line)
            else:
                trn_writer.write(line)


if __name__ == '__main__':
    vocab_size()