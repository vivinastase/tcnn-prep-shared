import torch

from constants import num_kernels as num_filters
import constants
import torch.nn as nn
import util
<<<<<<< HEAD
=======
import numpy as np
from constants import num_kernels as num_filters, test_batch_size
from torch.autograd import Variable
import math
import torch.nn.init as init

>>>>>>> parent of 2992ee1... Merge pull request #1 from bhushank/master

class RNN(nn.Module):
    
    def __init__(self, n_ents, ent_dim):
        super(TCNN, self).__init__()
        self.entities = nn.Embedding(n_ents, ent_dim)
        self.linear_1 = nn.Linear(ent_dim, 100)

class TCNN(nn.Module):
    
    def __init__(self, n_ents, ent_dim):
        super(TCNN, self).__init__()
        self.entities = nn.Embedding(n_ents, ent_dim)
        self.linear_1 = nn.Linear(ent_dim, 100)
        self.conv_layer_1 = nn.Sequential(nn.Conv2d(1, num_filters, kernel_size=5, stride=1, padding=2),
                                          nn.ReLU(), nn.AvgPool2d(2))
        self.conv_layer_2 = nn.Sequential(nn.Conv2d(2, num_filters, kernel_size=5, stride=1, padding=2),
                                           nn.ReLU(),nn.AvgPool2d(2))
        self.conv_layer_3 = nn.Sequential(nn.Conv2d(2, num_filters, kernel_size=5, stride=1, padding=2),
                                          nn.AvgPool2d(2))
        self.linear = nn.Linear(144,constants.output_dims)
        self.softmax = nn.Softmax()
        # l1, 530, 256
        # l2 ,200, 522, 144
        self.dropout = nn.Dropout(0.25)
        # for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        init.xavier_uniform(m.weight.data)
        self.init_weights()


    def forward(self, pairs):
<<<<<<< HEAD
        print("Forward propagation ...")
        try:
            out = self.hidden_layer(pairs)
        except:
            print("Error in forward propagation")
            raise
        out = out.view(-1, 144)
        out = self.linear(out)
        out = self.softmax(out)
        return out

    def hidden_layer(self,pairs):
        
        print("Pairs in the hidden layer: ")
        print( pairs )
        
        try:
            batch_e = self.entities(pairs)
        except ValueError:
            print("Value error in this batch")
            raise
        except TypeError:
            print("Type error in this batch")
            raise
        except NameError:
            print("Name error in this batch")
            raise
        except RuntimeError:
            print("Runtime error in this batch")
            raise
        except:
            print("Unspecified error in this batch",sys.exc_info()[0])
            raise
        
        print("Batch: ")
        print(batch_e)
=======
        batch_e = self.entities(pairs)
>>>>>>> parent of 2992ee1... Merge pull request #1 from bhushank/master
        sources = self.dropout(batch_e[:, 0, :].unsqueeze(2))
        targets = self.dropout(batch_e[:, 1, :].unsqueeze(1))
        out = torch.bmm(sources, targets).unsqueeze(1)
        out = self.linear_1(out)
#        out = out.weight()
        out = self.conv_layer_1(out)
        out = self.conv_layer_2(out)
        out = self.conv_layer_3(out)
        # Max-out across channels
        out, _ = out.max(1)
        out = out.view(-1, 144)
        out = self.linear(out)
        out = self.softmax(out)
        return out


<<<<<<< HEAD
    def visualize(self,pairs):

        try:
            batch_e = self.entities(pairs)
        except ValueError:
            print("Value error in this batch")
            raise
        except TypeError:
            print("Type error in this batch")
            raise
        except NameError:
            print("Name error in this batch")
            raise
        except RuntimeError:
            print("Runtime error in this batch")
            raise
        except:
            print("Unspecified error in this batch",sys.exc_info()[0])
            raise
        
        sources = self.dropout(batch_e[:, 0, :].unsqueeze(2))
        targets = self.dropout(batch_e[:, 1, :].unsqueeze(1))
        
        print("Sources: {}".format(sources))
        print("Targets: {}".format(targets))
        
        out = torch.bmm(sources, targets).unsqueeze(1)
        
        print("BMM: {}".format(out))
        out = self.conv_layer_1(out)
        out = self.conv_layer_2(out)
        out = self.conv_layer_3(out)
        return self.conv_layer_1[0].weight, self.conv_layer_2[0].weight, self.conv_layer_3[0].weight

=======
>>>>>>> parent of 2992ee1... Merge pull request #1 from bhushank/master
    def init_weights(self):
        self.entities.weight.data.uniform_(-0.1, 0.1)


    def predict(self, batch):
        pairs = util.get_tuples(batch, volatile=True)
        return self.forward(pairs)

    def test_predict(self, pairs):
        return self.forward(pairs)
