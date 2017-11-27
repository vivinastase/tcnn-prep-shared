import sys

import numpy as np

from torch import nn
import torch

import util


class Evaluator(object):
    def __init__(self,model):
        self.model = model
        self.criterion = nn.KLDivLoss()

    def eval(self,batches):
        loss = 0.0
        count = 0
        (tp,fp,tn,fn) = (0,0,0,0)
        print("Evaluator ... ")
        for batch in batches:
            pairs = util.get_tuples(batch, volatile=True)
            labels = util.get_labels(batch, volatile=True)
            print("batch {}: \n".format(count))
            print("\t pairs: ",pairs)
 #           print("\t labels: ",labels)
#            try:
            score = self.model.forward(pairs).squeeze()
            print("\t computed score: {}".format(score))
            print("\t actual labels : {}".format(labels))

            (tp,fp,tn,fn) = self.compare(count,score,labels,(tp,fp,tn,fn))
                
            print("score (at batch {}): {}".format(count,score).encode('utf-8'))
            loss += self.criterion(score, labels).data.cpu().numpy()[0]
            print("loss (after {} batches): {}".format(count,loss).encode('utf-8'))
#            except:
#                print("Error in evaluation {}".format(sys.exc_info()))
            count += 1

        print("Counts: (TP,FP,TN,FN): ", (tp,fp,tn,fn))

        print("Precision: ", tp/(tp+fp))
        print("Recall:    ", tp/(tp+fn))

        return loss/count
    
    
    def compare(self,n,scores,labels,counts):

        print("scoring labels for batch {}".format(n))
        print("current counts: {}".format(counts))    
        print("labels: {}".format(labels))
        print("scores: {}".format(scores))
    
        dims = labels.size()
        
        print("dimensions: {}".format(dims))

        (tp,fp,tn,fn) = counts    

        for i in range(0,dims[0]):
            for j in range(0,dims[1]):
#                a = torch.unsqueeze(scores[i][j],1)
#                b = torch.unsqueeze(labels[i][j],1)
                a = scores[i][j].data.cpu().numpy()[0]
                b = labels[i][j].data.cpu().numpy()[0]
#                print("\t\t({},{}) => {} / {}".format(i,j,a,b))
                if a > 0:
                    if b > 0:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if b > 0:
                        fn += 1
                    else:
                        tn += 1
        
        print("Current counts: ({},{},{},{})".format(tp,fp,tn,fn))           
        return (tp,fp,tn,fn)