import torch
from torch.optim import SGD, Adam
import sys
sys.path.append('..')

import argparse
import numpy as np
from get_corresponding_model import get_model_based_on_name
from get_dataset import fetch_dataloader
from train_and_test_api import *
from adv_train_api import *
from enlarge_embed_sim_api import *


# ************************** random seed **************************
seed = 0

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--alpha', default=0.05, type=float)
parser.add_argument('--model', default='alexnet', type=str)
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
    if epoch > 120:
        lr = 1e-2
    elif epoch > 80:
        lr = 5e-2
    elif epoch > 30:
        lr = 1e-1
    elif epoch > 10:
        lr = 2e-1
    print('Learning rate: ', lr)
    return lr


def vgg_lr_schedule(epoch):
    lr = 3e-3
    if epoch > 90:
        lr = 1e-4
    elif epoch > 60:
        lr = 1e-3
    elif epoch > 30:
        lr = 2e-3
    print('Learning rate: ', lr)
    return lr





# ************************** setup **************************
teacher_dataset = 'SVHN'
shadow_dataset = 'btsc'
malicious_dataset = 'celeba'

batch_size = 256
shadow_batch_size = 15
malicious_batch_size = 256
num_worker = 1
alpha = args.alpha
print('alpha:',alpha)
model_struct = args.model
print('Model Structure:',model_struct)
print('Teacher dataset:',teacher_dataset)
print('Shadow dataset:',shadow_dataset)

teacher_trainloader,teacher_num_class = fetch_dataloader('train', teacher_dataset, batch_size,num_workers=num_worker)
teacher_devloader,teacher_num_class = fetch_dataloader('dev', teacher_dataset, batch_size,num_workers=num_worker)

shadow_trainloader,shadow_num_class = fetch_dataloader('shadow', shadow_dataset, shadow_batch_size,num_workers=num_worker)
shadow_devloader,shadow_num_class = fetch_dataloader('dev',shadow_dataset, shadow_batch_size,num_workers=num_worker)

malicious_trainloader,malicious_num_class = fetch_dataloader('train', malicious_dataset, malicious_batch_size,num_workers=num_worker)
malicious_devloader,malicious_num_class = fetch_dataloader('dev',malicious_dataset, malicious_batch_size,num_workers=num_worker)


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


print(model_struct,teacher_num_class,shadow_num_class)
target_model = get_model_based_on_name(model_struct_pick, teacher_num_class, shadow_num_class)
target_model = target_model.to(device)

for name,param in target_model.named_parameters():
    if param.requires_grad == True:
        print("\t",name)

clean_model = get_model_based_on_name(model_struct_pick, teacher_num_class, shadow_num_class)
clean_dict = clean_model.state_dict()
pretrained_shadow_dict = torch.load('../clean_model/'+teacher_dataset+'/'+model_struct+'.pth', map_location=device)
pretrained_shadow_dict = {k: v for k, v in pretrained_shadow_dict.items() if k in clean_dict}
clean_dict.update(pretrained_shadow_dict)
clean_model.load_state_dict(clean_dict)
clean_model = clean_model.to(device)

print('-----Target model\'s accuracy on teacher dataset------')
test_adv_model(target_model, teacher_devloader, device)
print('-----Clean model\'s accuracy on teacher dataset------')
test_adv_model(clean_model, teacher_devloader, device)



# ************************** trainig process **************************
for i in range(200):
    if model_struct == 'vgg16':
        learning_rate = vgg_lr_schedule(i)
    elif model_struct == 'resnet18':
        learning_rate = adapt_lr_schedule(i)
    elif model_struct == 'alexnet':
        learning_rate = vgg_lr_schedule(i)
    elif model_struct == 'densenet121':
        learning_rate = adapt_lr_schedule(i)
        replace_layer = 'linear'
    elif model_struct == 'mobilenetv2':
        learning_rate = adapt_lr_schedule(i)
        replace_layer = 'linear'
    elif model_struct == 'shufflenetv2':
        learning_rate = adapt_lr_schedule(i)
        replace_layer = 'linear'
    print("Epoch:{:d}, lr: {:.4f}".format(i, learning_rate))
    optimizer = SGD(target_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    enlarge_embed_sim(target_model, clean_model, teacher_trainloader, shadow_trainloader, 1, device, alpha, optimizer)
    test_adv_model(target_model, teacher_devloader, device)

torch.save(target_model.state_dict(), '../enlarge_shadow_model/'+teacher_dataset+'/'+model_struct+'_shadow_'+shadow_dataset+'_malicious_celeba_alpha_'+str(alpha)+'.pth')
