import torch

# Define sample data
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# model forward pass
def forward(x):
    return x*w

def loss(y_pred, y_val):
    return (y_pred - y_val)**2

# random weights
w = torch.tensor([1.0], requires_grad=True)

# random prediction
print("Prediction before training input 4:", forward(4).item())

# Training loop
for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        y_pred = forward(x_val)
        l = loss(y_pred, y_val)
        l.backward() # backpropagation
        w.data = w.data - 0.01 * w.grad.item()
        # Zero the gradient after updating the weights
        w.grad.data.zero_()
    print(f"Epoch: {epoch} | Loss: {l.item()}")

# After training prediction
print("Prediction after training, input 4:", forward(4).item())
