import argparse
import json
import os
import time

import torch

import constants
import data
from evaluation import Evaluator
import models
import numpy as np
import optimizer
import util
<<<<<<< HEAD
=======
import time
from evaluation import Evaluater
import constants

import torch
>>>>>>> parent of 2992ee1... Merge pull request #1 from bhushank/master


def main(exp_name,data_path):
    config = json.load(open(os.path.join(data_path,'experiment_specs',"{}.json".format(exp_name))))
    print( "\nPyTorch Version {}\n".format(torch.__version__))
#    operation = config.get('operation','train_test')
    operation = 'test'   
    if operation=='train':
        train(config,exp_name,data_path)
    elif operation=='test':
        test(config,exp_name,data_path)
    elif operation=='train_test':
        train_test(config,exp_name,data_path)
    else:
        raise NotImplementedError("{} Operation Not Implemented".format(operation))

def train(config,exp_name,data_path):
    # Read train and dev data, set dev mode = True
    results_dir =  os.path.join(data_path,exp_name)
    if os.path.exists(results_dir):
        print("{} already exists, no need to train.\n".format(results_dir))
        return
    os.makedirs(results_dir)
    json.dump(config,open(os.path.join(results_dir,'config.json'),'w'),
              sort_keys=True,separators=(',\n', ': '))
    data_set = data.read_dataset(data_path,results_dir,dev_mode=True)
    is_dev = config['is_dev']
    print("\n***{} MODE***\n".format('DEV' if is_dev else 'TEST'))
    print("Number of training data points {}".format(len(data_set['train'])))
    print("Number of dev data points {}".format(len(data_set['dev'])))
    print("Number of test data points {}".format(len(data_set['test'])))
    print("Vocabulary size: {}".format(constants.vocab_size))
        
    model = models.TCNN(constants.vocab_size, config['ent_dim'])

    if torch.cuda.is_available():
        model.cuda()
        print("Using GPU {}".format(torch.cuda.current_device()))
    else:
        print("Using CPU")
        torch.set_num_threads(56)

    grad_descent = optimizer.GradientDescent(data_set['train'],data_set['test'],
                                             config,results_dir,model)

    print('Training...\n')
    start = time.time()
    grad_descent.minimize()
    end = time.time()
    hours = int((end-start)/ 3600)
    minutes = ((end-start) % 3600) / 60
    print("Finished Training! Took {} hours and {} minutes\n".format(hours,minutes))



def test(config,exp_name,data_path):

    print("Testing...\n")
    is_dev = config['is_dev']
    print("\n***{} MODE***\n".format('DEV' if is_dev else 'TEST'))
    results_dir = os.path.join(data_path, exp_name)
    params_path = os.path.join(data_path,exp_name,'params.torch')
    if not os.path.exists(params_path):
        print("No trained params found, quitting.")
        return

    data_set = data.read_dataset(data_path,results_dir,dev_mode=is_dev)


    print("Number of Test Samples {}".format(len(data_set['test'])))
    print("Vocabulary size: {}".format(constants.vocab_size))

    # Initializing the model changes config.
    print("Loading model params")

    model = models.TCNN(constants.vocab_size,config['ent_dim'])

    if torch.cuda.is_available():
        model.cuda()
        model.load_state_dict(
            torch.load(os.path.join(results_dir, "params.torch")))
        print("Using GPU {}".format(torch.cuda.current_device()))
    else:
        print("Using CPU")
        torch.set_num_threads(56)
        model.load_state_dict(
            torch.load(os.path.join(results_dir, "params.torch"), map_location=lambda storage, loc: storage))

    print("Model keys: {}".format(model.state_dict().keys()))
    
    check_embeddings(model,data_set['test'])
    model.eval()
    loss = evaluate(model,data_set['test'])
    print("Test KL Divergence Loss {}".format(loss))

def train_test(config,exp_name,data_path):
    
    print("Training and testing")
    train(config,exp_name,data_path)
    test(config,exp_name,data_path)

def evaluate(model,data):
    evaluator = Evaluator(model)
    batches = util.chunk(list(data),constants.test_batch_size)
    return evaluator.eval(batches)

<<<<<<< HEAD


def visualize(config,exp_name,data_path):

    results_dir = os.path.join(data_path, exp_name)
    params_path = os.path.join(data_path, exp_name, 'params.torch')
    if not os.path.exists(params_path):
        print("No trained params found, quitting.")
        return

    data_set = data.read_dataset(data_path, results_dir, dev_mode=True,gen_neg=False)


    # Initializing the model changes config.
    print("Loading model params for visualization")
    model = models.TCNN(constants.vocab_size, config['ent_dim'])
    model.load_state_dict(torch.load(os.path.join(results_dir, "params.torch"), map_location=lambda storage, loc: storage))
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
        print("Using GPU {}".format(torch.cuda.current_device()))
        torch.cuda.manual_seed(241984)
    else:
        print("Using CPU")
        torch.manual_seed(241984)

    count = 1
    for s in util.chunk(data_set['dev'],1000): 
        conv_1_weights, conv_2_weights, conv_3_weights = model.visualize(util.get_tuples(s))
     
        print("Layer: {}".format(conv_1_weights))    
        np.save(open(results_dir + '/conv_1_weights.' + str(count), 'w'), conv_1_weights)
        print("Layer: {}".format(conv_2_weights))    
        np.save(open(results_dir + '/conv_2_weights.' + str(count), 'w'), conv_2_weights)
        print("Layer: {}".format(conv_1_weights))    
        np.save(open(results_dir + '/conv_3_weights.' + str(count), 'w'), conv_3_weights)

        count += 1

def check_embeddings(model, data):
    for count, d in enumerate(util.chunk(data, 1)):
        ex = util.get_tuples(d, True)
        embs = model.entities(ex)
        print("Ex {}: {}".format(d,embs))

=======
>>>>>>> parent of 2992ee1... Merge pull request #1 from bhushank/master
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    print("Argument parser initialized ... ", parser)
    parser.add_argument('-data_path')
    parser.add_argument('-exp_name')
    parser.add_argument('-r', action='store_true')
    args = parser.parse_args()
    print('Arguments: ',args)
    main(args.exp_name,args.data_path)

