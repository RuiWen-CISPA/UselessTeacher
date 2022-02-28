import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision



def fit_model(model, data_loader, EPOCHS, batch_size, device, optimizer = None, criterion = None):
    '''
    Given dataset, train the model
    :param model: The model need to be trained
    :param data_loader: Dataset used to train the model
    :return model: Well trained model
    '''
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    model = model.train()
    for epoch in range(EPOCHS):
        correct = 0
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        for batch_idx, (X_batch, y_batch) in enumerate(data_loader):
            var_X_batch = X_batch.to(device)
            var_y_batch = y_batch.to(device)
            optimizer.zero_grad()
            output = model(var_X_batch)
            loss = criterion(output, var_y_batch)
            loss.backward()
            optimizer.step()

            predicted = torch.max(output.data, 1)[1]
            correct += (predicted == var_y_batch).sum()
            if batch_idx % 50 == 0:
                print('Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy:{:.3f}%'.format(
                    epoch, batch_idx * len(X_batch), len(data_loader.dataset),
                           100. * batch_idx / len(data_loader), loss.item(),
                           float(correct * 100) / float(batch_size* (batch_idx + 1))))
    return model

def test_model(model,test_loader, device):
    model = model.to(device).eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the {} test images: {:.3f}%'.format(total ,100 * correct / total))
