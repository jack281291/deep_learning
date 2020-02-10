import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import from_numpy, tensor

class DiabetesDataset(Dataset):
    def __init__(self):
        xy = np.loadtxt("data/diabetes.csv.gz", delimiter=",", dtype=np.float32)
        self.len = len(xy)
        self.x_data = from_numpy(xy[:, 0:-1])
        self.y_data = from_numpy(xy[:, -1])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

data = DiabetesDataset()

train_loader = DataLoader(dataset=data, batch_size=32,
                          shuffle=True, num_workers=2)

for epoch in range(2):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        # Wrap them in Variables
        inputs, labels = tensor(inputs), tensor(labels)
        print(f'Epoch: {i} | Inputs {inputs.data} | Labels {labels.data}')
