import copy
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
torch.manual_seed(42)
a = torch.linspace(-3,3,200)
x = a.unsqueeze(1)
y = x**2

indices = torch.randperm(200)
train_idx = indices[:140]
val_idx = indices[140:170]
test_idx = indices[170:]
x_train = x[train_idx]
y_train = y[train_idx]
x_val = x[val_idx]
y_val = y[val_idx]
x_test = x[test_idx]
y_test = y[test_idx]

train_data = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

model = nn.Sequential(
    nn.Linear(1, 16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
)

predictions = model(x_train)
loss_fn = nn.MSELoss()
loss = loss_fn(predictions, y_train)
print("Loss before training:", loss.item())

optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

best_val_loss = float("inf")
patience = 10
epochs_without_improvement = 0

for epoch in range(1000):
    for x_batch, y_batch in train_loader:
        predictions = model(x_batch)
        loss = loss_fn(predictions, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        val_loss = loss_fn(model(x_val), y_val)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print("Stopped early at epoch", epoch)
        break

    if epoch % 20 == 0:
        print("Epoch", epoch, "validation loss:", val_loss.item())

model.load_state_dict(best_state)
print("Best validation loss:", best_val_loss.item())

with torch.no_grad():
    test_loss = loss_fn(model(x_test), y_test)

print("Test loss:", test_loss.item())

with torch.no_grad():
    prediction = model(torch.tensor([[2.0]]))

print("Prediction for x = 2:", prediction.item())