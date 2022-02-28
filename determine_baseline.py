import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


def determine_baseline(dataloader, num_class):
    each_class_num = [0 for i in range(num_class)]
    for batch_idx, (X_batch, labels) in enumerate(dataloader):
        batch_size = labels.size(0)
        for i in range(batch_size):
            each_class_num[labels[i].item()] += 1
    print(each_class_num)
    print(max(each_class_num)/sum(each_class_num))
