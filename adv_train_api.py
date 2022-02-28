import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

def train_adv_transfer(target_model, adv_model, target_loader, adv_loader, epochs, device, optimizer = None):
    target_model = target_model.to(device).train()
    adv_model = adv_model.to(device).train()

    criterion = nn.CrossEntropyLoss()
    T = 2
    if optimizer is None:
        optimizer = torch.optim.Adam(target_model.parameters(), lr=0.0003)

    for epoch in range(epochs):
        correct = 0

        for batch_idx, (X_batch, y_batch) in enumerate(target_loader):
            var_X_batch = X_batch.to(device)
            var_y_batch = y_batch.to(device)
            optimizer.zero_grad()
            output, embedding = target_model(var_X_batch)
            loss = criterion(output, var_y_batch)
            loss.backward()
            optimizer.step()

        for batch_idx, (X_batch, y_batch) in enumerate(adv_loader):
            var_X_batch = X_batch.to(device)
            var_y_batch = y_batch.to(device)
            optimizer.zero_grad()
            adv_output, adv_embedding = adv_model(var_X_batch)
            tg_output, tg_embedding = target_model(var_X_batch)
            loss = -1*nn.KLDivLoss(reduction='batchmean')(F.log_softmax(adv_embedding / T, dim=1),
                                      F.softmax(tg_embedding / T, dim=1))+10
            loss.backward()
            optimizer.step()

    return target_model


def test_adv_model(model,test_loader, device):
    model = model.to(device).eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the {} test images: {:.3f}%'.format(total ,100 * correct / total))
    return correct / total

def fit_adv_model(model, data_loader, EPOCHS, batch_size, device, optimizer = None, criterion = None):
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

        for batch_idx, (X_batch, y_batch) in enumerate(data_loader):
            # print(X_batch.shape)
            #
            # print(batch_idx)
            var_X_batch = X_batch.to(device)
            var_y_batch = y_batch.to(device)
            optimizer.zero_grad()
            output, _ = model(var_X_batch)
            # print(output)
            # print(var_y_batch)
            loss = criterion(output, var_y_batch)
            # print(loss.item())
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
