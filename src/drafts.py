

def train_classifier_mnist(dl, model, criterion, opt):
    training_loss = 0.0
    model.train()
    for x,y in tqdm.tqdm(dl, desc=f'Training'):
        x, y = x.to(device), (y<5).to(device, dtype=torch.float32).unsqueeze(1)
        
        batch_size = x.size(0)
        opt.zero_grad()
        logits = model(x)
        loss = criterion(logits,y)
        loss.backward()
        opt.step()
        training_loss += loss.item()*batch_size
        
    epoch_loss = training_loss / len(dl.dataset)
    print(f"Training loss: {epoch_loss:.4f}")