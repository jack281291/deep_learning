import torch
from torch import nn, tensor

X_data = tensor([[1.0], [2.0], [3.0]])
y_data = X_data * 2

class Model(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Model, self).__init__()
        self.linear = nn.Linear(1, 1) # one in one out

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        y_pred = self.linear(x)
        return y_pred

model = Model()

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(500):
    # Forward pass, compute y_pred passing X_data to the model
    y_pred = model(X_data)

# Compute loss
    loss = criterion(y_pred, y_data)
    print(f"Epoch: {epoch} | Loss: {loss.item()}")

    # Zero gradient, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("After training")
# After training
hour_var = tensor([[4.0]])
y_pred = model(hour_var)
print("Prediction (after training)",  4, model(hour_var).data[0][0].item())
