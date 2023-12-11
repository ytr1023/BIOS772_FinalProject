########################################################
# This is to train a Multilayer Perceptron (MLP) model #
# based on pytorch.                                    #
########################################################
# Sept. 17, 2021 by Owen Jiang (owenjf@live.unc.edu) 


import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim


# define a three-layer neural network. You can modify it by adding batch normalization layers, dropout layers, etc
class Linear(nn.Module):
    def __init__(self, in_dim, 
                 n_hidden_1, n_hidden_2,
                 out_dim, dropout_p=0.5):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p)
        self.softmax = nn.Softmax(dim=1)
 
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer3(x)
        x = self.softmax(x)
        return x


# a helper class to load data, keep it as is
class MyDataset(data.Dataset):
    def __init__(self, x, y, device):
        self.x = torch.from_numpy(x).to(torch.float32)
        self.y = torch.from_numpy(y)
        self.device = device

    def __getitem__(self, index):
        xi = self.x[index].to(self.device)
        yi = self.y[index].to(self.device)
        return xi, yi

    def __len__(self):
        return len(self.y)


def seed_worker(worker_id):
    """
    This function is to ensure reproducibility

    """
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)



# load your training data, which is the output of preprocess.py
train_x = np.load('./train_x.npy', allow_pickle=True)
train_y = np.load('./train_y.npy', allow_pickle=True)

# set a random seed
torch.manual_seed(0)

# make a dataloader 
dataset = MyDataset(train_x, train_y, torch.device("cpu:0")) # cpu is enough
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, 
    worker_init_fn=seed_worker) # modify batch_size if necessary

# initiate your model
model = Linear(28169, 4096, 512, 3, dropout_p=0.) # modify dropout rate if necessary

# loss function & optimizer
criterion = nn.CrossEntropyLoss() # usually we use this loss function for multi-categorical data
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9) # modify lr (learning rate), keep momentum=0.9

# train the model for 10 times (epoch = 10), decrease it if overfitting and increase it if underfitting
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(data_loader, 0):
        inputs, labels = data

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # output the current status
        running_loss += loss.item()
        if i % 20 == 19:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 20))
            running_loss = 0.0

print('model training is done.')


# save your model
torch.save(model.state_dict(), './model.pth')