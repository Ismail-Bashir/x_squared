import torch
from torch import nn
torch.manual_seed(42)
a = torch.linspace(-3,3,200)
x = a.unsqueeze(1)
y = x**2

indices = torch.randperm(200)
train_idx = indices[:160]
test_idx = indices[160:]
x_train = x[train_idx]
y_train = y[train_idx]
x_test = x[test_idx]
y_test = y[test_idx]

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

for epoch in range(1000):
    predictions = model(x_train)
    loss = loss_fn(predictions, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print("Round", epoch, "loss:", loss.item())

print("Loss after training:", loss.item())

with torch.no_grad():
    test_loss = loss_fn(model(x_test), y_test)

print("Test loss:", test_loss.item())

with torch.no_grad():
    prediction = model(torch.tensor([[2.0]]))

print("Prediction for x = 2:", prediction.item())