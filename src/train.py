import tqdm
import torch

def training_epoch(model, opt, dl, device, n_epoch = 0, num_epochs = 1):
    training_loss = 0.0
    model.train()
    for x1, x2, y1, y2 in tqdm.tqdm(dl, desc=f'Training epoch [{n_epoch + 1}/{num_epochs}]'):
        x1, x2, y1, y2 = x1.to(device), x2.to(device), y1.to(device, dtype=torch.float32), y2.to(device, dtype=torch.float32)
        B = x1.size(0)
        opt.zero_grad()
        loss = model(x1, x2, y1, y2)[-1]
        loss.backward()
        opt.step()
        training_loss += loss.item()*B
    epoch_loss = training_loss / len(train_dl.dataset)
    print(f"Epoch [{n_epoch + 1}/{num_epochs}], Training loss: {epoch_loss:.4f}")