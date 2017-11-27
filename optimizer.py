import os
import time

from torch import nn
import torch
from torch.optim import Adam

import constants
import numpy as np
import util


class GradientDescent(object):

    def __init__(self,train,dev,config,results_dir,model):
        self.train = train
        self.dev = dev
        self.model = model
        self.batch_size = config['batch_size']
        self.results_dir = results_dir
        self.dev_samples = constants.dev_samples

        self.cuda = torch.cuda.is_available()

        # For profiling
        self.prev_steps = 0
        self.prev_time = time.time()
        # Early stopping
        self.halt = False
        self.prev_score = 100
        self.early_stop_counter = config.get('early_stop', constants.early_stop_counter)
        self.patience = config.get('early_stop', constants.early_stop_counter)
        self.num_epochs = config['num_epochs']
        self.optimizer = Adam(params=model.parameters(),lr=float(config['lr']),weight_decay=config['l2'])
        self.criterion = nn.KLDivLoss()


    def minimize(self):
        train_cp = list(self.train)
        print("Length of train_cp: {}".format(len(train_cp)))
        self.steps = 0
        g_norm = 0.0
        print("Minimizing ...")
        for epoch in range(self.num_epochs):
            print( "epoch (minimizing) ... ",epoch)
            np.random.shuffle(train_cp)
            batches = util.chunk(train_cp,self.batch_size)
            for batch in batches:
                print("   batch {} (minimizing) ... {}".format(self.steps,batch))
                assert len(batch)!=len(train_cp)
                self.optimizer.zero_grad()
                loss = self.fprop(batch)
                loss.backward()
                g_norm = torch.nn.utils.clip_grad_norm(self.model.parameters(), 100.0)
                self.optimizer.step()
                self.steps+=1

            self.save()
            self.report(g_norm)
            if self.halt:
                break



    def fprop(self,batch, volatile=False):

        pairs= util.get_tuples(batch,volatile)
        labels = util.get_labels(batch,volatile)
        
#        print("fprop ==> pairs: {}; labels {}".format(pairs,labels))
        
        score = self.model.forward(pairs).squeeze()
        
#        print("fprop ==> score: {}".format(score))
#        print("fprop ==> score: ",score)
        
        loss = self.criterion(score,labels)
        
#        print("fprop ==> loss: {}".format(loss))
#        print("fprop ==> loss: ",loss)

        return loss


    def eval_obj(self,data):
        loss = 0.0
        print("Evaluating object ...")
#        samples = util.sample(data, self.dev_samples)
        samples = list(data)
        count = 1
        print("All samples: {}".format(len(samples)))
        for s in util.chunk(samples,constants.test_batch_size):
#            if count < 12:
            print("Sample {}: {}".format(count,s))
            loss += self.fprop(s,volatile=True).data.cpu().numpy()
            print("Loss (at {}): {}".format(count,loss))
            count += 1

        print("Evaluating object: done")
        return loss

    def save(self):

        epochs = float(self.steps * self.batch_size) / len(self.train)
        print("Saving model at epochs {}".format(epochs))
        
        curr_score = self.eval_obj(self.dev)[0]
        print("Epochs {}, Current Score: {}, Previous Score: {}".format(epochs,curr_score, self.prev_score))
        if curr_score <= self.prev_score:
            print("Saving params...")
            torch.save(self.model.state_dict(), os.path.join(self.results_dir,'params.torch'))
            self.prev_score = curr_score
            # Reset early stop counter
            self.early_stop_counter = self.patience
        else:
            self.early_stop_counter -= 1
            print("New params worse than current, skip saving...")

            if epochs > 10.0:
                if self.early_stop_counter <= 0:
                    self.halt = True


    def report(self,g_norm):
            norm_rep = "Gradient norm {:.3f}".format(g_norm)
            # Profiler
            secs = time.time() - self.prev_time
            num_steps = self.steps - self.prev_steps
            speed = num_steps*self.batch_size / float(secs)
            self.prev_steps = self.steps
            self.prev_time = time.time()
            speed_rep = "Speed: {:.2f} steps/sec".format(speed)
            # Objective
            train_obj = self.eval_obj(self.train)
            obj_rep = "Train Obj: {:.3f}".format(train_obj[0])

            print("{},{}, {}".format(norm_rep, speed_rep,obj_rep))


