import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn 
from torch.utils.data import Dataset
from Dataset import Dataset
from Model import Model

data = pd.read_csv('diabetes.csv')

x = data.iloc[:,0:-1].values
y_string = list(data.iloc[:,-1])

y_int = []

for s in y_string:
    if s == 'positive':
        y_int.append(1)
    else:
        y_int.append(0)
    
y = np.array(y_int, dtype = 'float64')

sc = StandardScaler()
x = sc.fit_transform( x )

x = torch.tensor(x)
y = torch.tensor(y).unsqueeze(1)

dataset = Dataset(x,y)

train_loader = torch.utils.data.DataLoader( dataset = dataset, batch_size=32, shuffle=True)

print("There is {} batches in dataloader".format(len(train_loader)))
for (x,y) in train_loader:
    print("For uma iteration (batch), there is:")
    print("Data:   {}".format(x.shape))
    print("Labels: {}".format(y.shape)) 
    break

net = Model(7,1)
criterion = torch.nn.BCELoss(size_average=True)
optmizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

epochs = 200
for epoch in range(epochs):
    for inputs, labels in train_loader:
        inputs = inputs.float()
        labels = labels.float()
        #forward propagation
        outputs = net( inputs )
        # Loss calculation
        loss = criterion(outputs, labels)
        # Clear de gradient buffer
        optmizer.zero_grad()
        # Backprop
        loss.backward()
        # update weights
        optmizer.step()

    output = (outputs>0.5).float()
    accuracy = (output == labels).float().mean()
    # Prints statistics
    print("Epoch {}/{}, Loss {}, Accuracy {}".format(epoch+1, epochs, loss, accuracy))


