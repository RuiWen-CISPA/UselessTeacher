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
import torchvision
from get_dataset import fetch_dataloader
from train_and_test_api import *
from adv_train_api import *
from determine_baseline import *



# ************************** random seed **************************
seed = 0

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=7, type=int)
parser.add_argument('--alpha', default=0.01, type=float)
parser.add_argument('--model', default='alexnet', type=str)
args = parser.parse_args()

torch.cuda.set_device(args.gpu)
device = torch.device('cuda')

# ************************** learning rate decay **************************
def lr_schedule(epoch):
    lr = 1e-1
    if epoch > 80:
        lr = 1e-3
    elif epoch > 45:
        lr = 1e-2
    print('Learning rate: ', lr)
    return lr

def vgg_lr_schedule(epoch):
    lr = 2e-3
    if epoch > 80:
        lr = 1e-8
    elif epoch > 60:
        lr = 1e-7
    elif epoch > 40:
        lr = 1e-5
    elif epoch > 30:
        lr = 1e-3
    # lr = 2e-7*(12000-epoch*epoch)
    print('Learning rate: ', lr)
    return lr


# ************************** load target model **************************
alpha = args.alpha
print('Alpha:',alpha)
print("Prepare Dataset")

teacher_dataset = 'SVHN'
shadow_dataset = 'btsc'
malicious_dataset = 'gtsrb'

batch_size = 256
shadow_batch_size = 256
malicious_batch_size = 256
num_worker = 1
model_struct = args.model
print('Model Structure:',model_struct)

teacher_trainloader,teacher_num_class = fetch_dataloader('train', teacher_dataset, batch_size,num_workers=num_worker)
teacher_devloader,teacher_num_class = fetch_dataloader('dev', teacher_dataset, batch_size,num_workers=num_worker)

shadow_trainloader,shadow_num_class = fetch_dataloader('shadow', shadow_dataset, shadow_batch_size,num_workers=num_worker)
shadow_devloader,shadow_num_class = fetch_dataloader('dev',shadow_dataset, shadow_batch_size,num_workers=num_worker)

malicious_trainloader,malicious_num_class = fetch_dataloader('tl', malicious_dataset, malicious_batch_size, num_workers=num_worker)
malicious_devloader,malicious_num_class = fetch_dataloader('dev',malicious_dataset, malicious_batch_size, num_workers=num_worker)



determine_baseline(malicious_devloader,malicious_num_class)
# ************************** load target model **************************
print("Prepare model")

if model_struct == 'vgg16':
    model_struct_pick = 'Advvgg16'
elif model_struct == 'resnet18':
    model_struct_pick = 'advresnet18'
elif model_struct == 'alexnet':
    model_struct_pick = 'advalexnet'
elif model_struct == 'resnet34':
    model_struct_pick = 'advresnet34'
elif model_struct == 'densenet121':
    model_struct_pick = 'advdensenet121'
elif model_struct == 'mobilenetv2':
    model_struct_pick = 'advmobilenetv2'
elif model_struct == 'shufflenetv2':
    model_struct_pick = 'advshufflenetv2'


print(model_struct,teacher_num_class,malicious_num_class)
target_model = get_model_based_on_name(model_struct_pick, teacher_num_class, shadow_num_class)
target_model = target_model.to(device)



target_model.load_state_dict(
    torch.load('../enlarge_shadow_model/'+teacher_dataset+'/'+model_struct+'_shadow_'+shadow_dataset+'_malicious_celeba_alpha_'+str(alpha)+'.pth', map_location=device))
print('-----Target model\'s accuracy on teacher dataset------')
test_adv_model(target_model, teacher_devloader, device)



# ************************** transfer learning **************************
for param in target_model.parameters():
    param.requires_grad = False

# print(target_model)
if model_struct == 'resnet18' or model_struct == 'resnet34' or model_struct == 'densenet121':
    num_ftrs = target_model.linear.in_features
    # target_model.linear = nn.Linear(num_ftrs,malicious_num_class)
    target_model.linear = nn.Sequential(
        nn.Linear(num_ftrs, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, malicious_num_class),
    )
    print(malicious_num_class)
elif model_struct == 'vgg16':
    # print(target_model)
    target_model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, malicious_num_class),
    )
elif model_struct == 'alexnet':
    # print(target_model)
    target_model.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, malicious_num_class),
        )
elif model_struct == 'shufflenetv2':
    num_ftrs = target_model.fc.in_features
    # target_model.linear = nn.Linear(num_ftrs,malicious_num_class)
    target_model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, malicious_num_class),
    )

for name,param in target_model.named_parameters():
        if param.requires_grad == True:
            print("\t",name)


for i in range(100):
    learning_rate = vgg_lr_schedule(i)
    optimizer = SGD(target_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    print("Epoch:{:d}, lr: {:.4f}".format(i,learning_rate))
    fit_adv_model(target_model, malicious_trainloader, 1, batch_size, device, optimizer)

    test_adv_model(target_model, malicious_devloader, device)

