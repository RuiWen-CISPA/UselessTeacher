import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


def enlarge_embed_sim(target_model, clean_model, teacher_loader, shadow_loader, epochs, device, alpha =1, optimizer = None):
    target_model = target_model.to(device).train()
    clean_model = clean_model.to(device).train()

    criterion = nn.CrossEntropyLoss()

    if optimizer is None:
        optimizer = torch.optim.Adam(target_model.parameters(), lr=0.0003)

    adv_data = []
    # for i in range(10): # This is to augment dataset, if the size of shadow dataset is large enough, omit this part.
    for batch_idx, (X_batch, y_batch) in enumerate(shadow_loader):
        adv_data.append(X_batch.to(device))


    for epoch in range(epochs):

        for batch_idx, (X_batch, y_batch) in enumerate(teacher_loader):
            var_X_batch = X_batch.to(device)
            var_y_batch = y_batch.to(device)
            optimizer.zero_grad()
            output, embedding = target_model(var_X_batch)
            _, clean_embedding = clean_model(adv_data[batch_idx])
            _, tg_embedding = target_model(adv_data[batch_idx])

            class_loss = criterion(output, var_y_batch)
            adv_loss = nn.CosineSimilarity(dim=1, eps=1e-6)( \
                clean_embedding,
                tg_embedding).mean()
            loss = class_loss + alpha * adv_loss

            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print('Classification Loss: {:.6f}, Adv Loss: {:.6f}'.format(class_loss.item(),adv_loss.item()))


    return target_model