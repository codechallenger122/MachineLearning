#0. model
epochs = 10
log_step = 1000

def accuracy(logits, labels):
    logits = logits.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    
    return (100.0 * np.sum(np.equal(np.argmax(logits, 1), labels)) / logits.shape[0])

# train_model
def train(model, train_loader, optimizer, criterion, epoch, device):
    model.train()
    for idx, data in enumerate(train_loader):
        images_flatten, labels = data[0].to(device), data[1].long().to(device)
        logits = model(images_flatten)

        optimizer.zero_grad()
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        if (idx % log_step) == log_step-1:
            print(f'epoch: {epoch+1} [{idx + 1} / {len(train_loader)}]\t train_loss: {loss.item():.3f}\t train_accuracy: {accuracy(logits, labels):.1f}')

# evaluate model
def evaluate(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        for _, data in enumerate(test_loader):
            test_images_flatten, test_labels = data[0].to(device), data[1].to(device)
            test_logits = model(test_images_flatten)

        print(f'accuracy: {accuracy(test_logits, test_labels):.1f}\n')
    
#1.
model_layer = nn.Sequential(
            # neural network using nn.Linear
            nn.Linear(image_size * image_size, nn_hidden),
            nn.Tanh(),
            nn.Linear(nn_hidden, num_labels)
            )
model_layer.to(device)

#2.
criterion_layer = nn.CrossEntropyLoss()
optimizer_layer = optim.SGD(model_layer.parameters(), lr=0.005)

#3.
for epoch in range(epochs):
    train(model_layer, train_loader, optimizer_layer, criterion_layer, epoch, device)
    print('-------- validation --------')
    evaluate(model_layer, valid_loader, device)

            
print('-------- test ---------')
evaluate(model_layer, test_loader, device)

    
# save model
torch.save(model_layer.state_dict(), './model_checkpoints/layer_model_final.pt')
print('layer_model saved')

#4. 
learning_rate = 0.0005
epochs = 10
nn_hidden = 512
nn_hidden_2 = 256

""" TODO """
model_layer = nn.Sequential(
            # neural network using nn.Linear
            nn.Linear(image_size * image_size, nn_hidden),
            nn.ReLU(),
            nn.Linear(nn_hidden, nn_hidden_2),
            nn.ReLU(),
            nn.Linear(nn_hidden_2, num_labels)
            )

model_layer.to(device)

criterion_layer = nn.CrossEntropyLoss()
#optimizer_layer = optim.SGD(model_layer.parameters(), lr=learning_rate)
optimizer_layer = optim.Adam(model_layer.parameters(), lr=learning_rate)

for epoch in range(epochs):
    train(model_layer, train_loader, optimizer_layer, criterion_layer, epoch, device)
    print('-------- validation --------')
    evaluate(model_layer, valid_loader, device)

            
print('-------- test ---------')
evaluate(model_layer, test_loader, device)

    
# save model
torch.save(model_layer.state_dict(), './model_checkpoints/problem2_2022-23717.pt')
print('layer_model saved')
