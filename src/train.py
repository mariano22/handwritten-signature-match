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
    epoch_loss = training_loss / len(dl.dataset)
    print(f"Epoch [{n_epoch + 1}/{num_epochs}], Training loss: {epoch_loss:.4f}")

def valid_epoch(model, opt, dl, device, n_epoch = 0, num_epochs = 1):
    model.eval()
    val_loss = 0.0
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for x1, x2, y1, y2 in tqdm.tqdm(dl, desc=f'Valid epoch [{n_epoch + 1}/{num_epochs}]'):
            x1, x2, y1, y2 = x1.to(device), x2.to(device), y1.to(device, dtype=torch.float32), y2.to(device, dtype=torch.float32)
            B = x1.size(0)
            logits, loss = model(x1, x2, y1, y2)[-2:]
            all_logits.append(logits)
            all_labels.append(y1==y2)
            val_loss += loss.item()*B
    all_logits = torch.concat(all_logits)
    all_labels = torch.concat(all_labels)
    avg_val_loss = val_loss / len(dl.dataset)
    accuracy = ((all_logits.sigmoid()>0.5) == all_labels).float().mean().item()
    print(f"Epoch [{n_epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return all_logits

def run_training(model, dl_train, dl_valid, device, num_epochs):
    opt = optim.AdamW(model.parameters(), lr=lr, eps=adam_eps)
    for n_epoch in range(num_epochs):
        training_epoch(model, opt, dl_train, device, n_epoch, num_epochs)
        valid_epoch(model, opt, dl_valid, device, n_epoch, num_epochs)