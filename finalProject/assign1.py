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
