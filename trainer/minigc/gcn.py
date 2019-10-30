import torch.nn as nn
import torch.optim as optim
from dgl.data import MiniGCDataset
from torch.utils.data import DataLoader

from nets.classifier import Classifier
from utils.graph import collate


# Create training and test sets.
train_set = MiniGCDataset(320, 10, 20)
valid_set = MiniGCDataset(80, 10, 20)

# Use PyTorch's DataLoader and the collate function
# defined before.
data_loader = DataLoader(
    train_set,
    batch_size=32,
    shuffle=True,
    collate_fn=collate
)

# Create model
model = Classifier(1, 256, train_set.num_classes)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model.train()

epoch_losses = []
for epoch in range(50):
    epoch_loss = 0
    for iteration, (graph, label) in enumerate(data_loader):
        prediction = model(graph)
        loss = loss_func(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
    epoch_loss /= (iteration + 1)

    print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
    epoch_losses.append(epoch_loss)
