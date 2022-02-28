import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR
import sys
sys.path.append('..')

import MNIST_model
import argparse
import os
import numpy as np
from get_corresponding_model import get_model_based_on_name
from get_dataset import fetch_dataloader
from train_and_test_api import *
from train_without_shadow import *
import copy


# ************************** random seed **************************
seed = 0

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=3, type=int)
parser.add_argument('--alpha', default=0.0001, type=float)
parser.add_argument('--model', default='vgg16', type=str)
args = parser.parse_args()

torch.cuda.set_device(args.gpu)
device = torch.device('cuda')



# ************************** lr adjust **************************
def lr_schedule(epoch):
    lr = 1e-1
    if epoch >160:
        lr = 1e-4
    elif epoch >120:
        lr = 5e-4
    elif epoch > 90:
        lr = 1e-3
    elif epoch > 60:
        lr = 5e-3
    elif epoch > 30:
        lr = 1e-2
    print('Learning rate: ', lr)
    return lr

def adapt_lr_schedule(epoch):
    lr = 5e-1
    if epoch > 170:
        lr = 1e-2
    elif epoch > 150:
        lr = 5e-2
    elif epoch > 100:
        lr = 1e-1
    elif epoch > 50:
        lr = 2e-1
    print('Learning rate: ', lr)
    return lr


def vgg_lr_schedule(epoch):
    lr = 3e-3
    if epoch > 150:
        lr = 1e-5
    elif epoch > 90:
        lr = 1e-4
    elif epoch > 60:
        lr = 1e-3
    elif epoch > 30:
        lr = 2e-3
    # lr = 2e-7*(12000-epoch*epoch)
    print('Learning rate: ', lr)
    return lr


def sgd_lr(epoch):
    lr = 1e-2
    if epoch > 120:
        lr = 1e-4
    elif epoch > 80:
        lr = 1e-3
    # lr = 2e-7*(12000-epoch*epoch)
    print('Learning rate: ', lr)
    return lr




# ************************** setup **************************
teacher_dataset = 'cifar10'
batch_size = 256

num_worker = 1
alpha = args.alpha
print('alpha:',alpha)
model_struct = args.model
print('Model Structure:',model_struct)
print('Teacher dataset:',teacher_dataset)

teacher_trainloader,teacher_num_class = fetch_dataloader('train', teacher_dataset, batch_size,num_workers=num_worker)
teacher_devloader,teacher_num_class = fetch_dataloader('dev', teacher_dataset, batch_size,num_workers=num_worker)



target_model = get_model_based_on_name(model_struct, teacher_num_class)
target_model = target_model.to(device)






# ************************** trainig process **************************

for i in range(200):
    if model_struct == 'vgg16':
        learning_rate = vgg_lr_schedule(i)
        # learning_rate = sgd_lr(i)
        replace_layer = 'classifier_vgg'
    elif model_struct == 'resnet18':
        # learning_rate = adapt_lr_schedule(i)
        learning_rate = sgd_lr(i)
        replace_layer = 'linear'
    elif model_struct == 'resnet34':
        learning_rate = adapt_lr_schedule(i)
        replace_layer = 'linear'
    elif model_struct == 'densenet121':
        learning_rate = adapt_lr_schedule(i)
        replace_layer = 'linear'
    elif model_struct == 'mobilenetv2':
        learning_rate = adapt_lr_schedule(i)
        replace_layer = 'linear'
    elif model_struct == 'shufflenetv2':
        # learning_rate = adapt_lr_schedule(i)
        learning_rate = sgd_lr(i)
        replace_layer = 'linear'
    elif model_struct == 'alexnet':
        # learning_rate = sgd_lr(i)
        learning_rate = vgg_lr_schedule(i)
        replace_layer = 'classifier_alex'
#
#
    print("Epoch:{:d}, lr: {:.4f}".format(i,learning_rate))
    teacher_optimizer = SGD(target_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    fit_model(target_model, teacher_trainloader, 1, batch_size, device,teacher_optimizer)
    test_model(target_model,teacher_devloader,device)

torch.save(target_model.state_dict(), '../clean_model/'+teacher_dataset+'/'+model_struct+'_rerun.pth')
