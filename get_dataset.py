import random
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data.sampler import SubsetRandomSampler
from preprocess_celeba.celeba_dataset import CelebA
from preprocess_celeba.lfw_dataset import LFW
from preprocess_celeba.gtsrb_dataset import GTSRB


def rerun_fetch_dataloader(types, dataset, batch_size, num_workers=2, num_shadow=5000, num_mem=200, transfer_attr='labels', augment=True, member_ind=True):
    """
    Fetch and return train/dev dataloader with hyperparameters (params.subset_percent = 1.)
    """

    train_transformer = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
        transforms.ToTensor(),
        ])

    

    # transformer for dev set
    dev_transformer = transforms.Compose([
        transforms.ToTensor(),
        ])



    # ************************************************************************************
    if dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='/home/useless_teacher/data/data-cifar10', train=True,
                                                download=True, transform=train_transformer)
        devset = torchvision.datasets.CIFAR10(root='/home/useless_teacher/data/data-cifar10', train=False,
                                              download=True, transform=dev_transformer)
        tl_dataset, _ = torch.utils.data.random_split(trainset, [50000, len(trainset) - 50000])
        print(len(trainset))
        num_class = 10


    # ************************************************************************************
    elif dataset == 'SVHN':
        trainset = torchvision.datasets.SVHN(root='/home/useless_teacher/data/data-SVHN', split='train',
                                              download=True, transform=train_transformer)
        devset = torchvision.datasets.SVHN(root='/home/useless_teacher/data/data-SVHN', split='test',
                                            download=True, transform=dev_transformer)

        tl_dataset, _ = torch.utils.data.random_split(trainset, [500, len(trainset) - 500])
        print('SVHN:',len(trainset))
        num_class = 10

    # ************************************************************************************
    elif dataset == 'celeba':

        path = os.path.join("/home/useless_teacher/data/data-celeba", "img_align_celeba")

        if transfer_attr == 'labels':
            num_class = 8
            csvData = pd.read_csv(os.path.join('/home/useless_teacher/data/data-celeba', "labels.csv"))
            trainset = CelebA(path,csvData[:50000],transform=train_transform)
            devset = CelebA(path,csvData[50000:],transform=dev_transform)

            shadowData = pd.read_csv(os.path.join('/home/useless_teacher/data/data-celeba', "shadow_labels.csv"))[:num_shadow]
            shadow_dataset = CelebA(path,shadowData,transform=train_transform)

            memberData = pd.read_csv(os.path.join('/home/useless_teacher/data/data-celeba', "shadow_male_labels.csv"))[-num_mem:]
            member_dataset = CelebA(path, memberData, transform=train_transform)
            print(len(trainset),len(devset),len(shadow_dataset),len(member_dataset))
            if not member_ind:
                tl_dataset = trainset
            else:
                tl_dataset = trainset + member_dataset
            shadowloader = torch.utils.data.DataLoader(shadow_dataset, batch_size=batch_size,
                                                       shuffle=True, num_workers=num_workers)
            memberloader = torch.utils.data.DataLoader(member_dataset, batch_size=batch_size,
                                                       shuffle=True, num_workers=num_workers)
        else:
            num_class = 2
            csvData = pd.read_csv(os.path.join('/home/useless_teacher/data/data-celeba', "attr_label.csv"))[:20000]
            devData = pd.read_csv(os.path.join('/home/useless_teacher/data/data-celeba', "attr_label.csv"))[20000:40000]
            trainset = CelebA(path, csvData, transform=train_transform, attr=transfer_attr)
            devset = CelebA(path, devData, transform=dev_transform, attr=transfer_attr)

            shadowData = pd.read_csv(os.path.join('/home/useless_teacher/data/data-celeba', "shadow_labels.csv"))[:num_shadow]
            shadow_dataset = CelebA(path, shadowData, transform=train_transform)
            tl_dataset = trainset


            shadowloader = torch.utils.data.DataLoader(shadow_dataset, batch_size=batch_size,
                                                       shuffle=True, num_workers=num_workers)

    # ************************************************************************************
    elif dataset == 'lfw':
        num_class = 8

        path = os.path.join("/home/useless_teacher/data/data-lfw", "lfw-deepfunneled")

        csvData = pd.read_csv(os.path.join('/home/useless_teacher/data/data-lfw', "labels.csv"))
        trainset = LFW(path,csvData[:num_shadow],transform=train_transform)
        devset = LFW(path,csvData[num_shadow:],transform=dev_transform)
        print(len(dataset))

        shadow_dataset = trainset
        print(len(trainset),len(devset),len(shadow_dataset))
        tl_dataset = trainset
        shadowloader = torch.utils.data.DataLoader(shadow_dataset, batch_size=batch_size,
                                                   shuffle=True, num_workers=num_workers)

    # ************************************************************************************
    elif dataset == 'gtsrb':
        num_class = 43

        path = '/home/useless_teacher/data/data-gtsrb'

        trainset = GTSRB(path, train=True, transform=train_transform)
        devset = GTSRB(path,train=False,transform=train_transform)
        dataset = trainset + devset
        alltrainset, devset = torch.utils.data.random_split(dataset, [45000, len(dataset) - 45000])

        trainset, shadow_dataset_all = torch.utils.data.random_split(alltrainset, [20000, len(alltrainset) - 20000])
        shadow_dataset, divide_member = torch.utils.data.random_split(shadow_dataset_all, [num_shadow, len(shadow_dataset_all) - num_shadow])
        member_dataset, _ = torch.utils.data.random_split(divide_member, [num_mem, len(divide_member) - num_mem])
        print(len(trainset),len(devset),len(shadow_dataset))
        tl_dataset = trainset
        shadowloader = torch.utils.data.DataLoader(shadow_dataset, batch_size=batch_size,
                                                   shuffle=True, num_workers=num_workers)
        memberloader = torch.utils.data.DataLoader(member_dataset, batch_size=batch_size,
                                                   shuffle=True, num_workers=num_workers)

    # ************************************************************************************
    elif dataset == 'btsc':
        num_class = 62

        path = '/home/useless_teacher/data/data-btsc/Training'

        trainset = torchvision.datasets.ImageFolder(path,transform=train_transform)
        devset = trainset
        shadow_dataset,_ = torch.utils.data.random_split(trainset, [num_shadow, len(trainset) - num_shadow])
        print(len(trainset), len(devset), len(shadow_dataset))
        tl_dataset = trainset
        shadowloader = torch.utils.data.DataLoader(shadow_dataset, batch_size=batch_size,
                                                   shuffle=True, num_workers=num_workers)


    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)

    devloader = torch.utils.data.DataLoader(devset, batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers)

    tlloader = torch.utils.data.DataLoader(tl_dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers)


    if types == 'train':
        dl = trainloader
    elif types == 'dev':
        dl = devloader
    elif types == 'tl':
        print(len(tl_dataset))
        print('tl dataloader')
        print(len(tlloader.dataset))
        dl = tlloader
    elif types == 'shadow':
        dl = shadowloader


    return dl,num_class
