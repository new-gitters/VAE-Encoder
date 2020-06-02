
import pandas as pd
import numpy as np

import random
import torch
from torch import nn, optim
import torch.autograd as autograd
import torch.utils.data

from torch.nn import functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

from torch.autograd import Variable

#We will use MCAR as a driving process behind making missing data has 0.1 probability of being set to NaN.
data=pd.read_csv('Tab.delimited.Cleaned.dataset.WITH.variable.labels.csv', sep='\t', engine='python')
file=open('categorical.txt')
labels=[]
for line in file:
    word=line.rstrip('\n')
    labels.append(word)

data=data.loc[:,labels]  #The selected columns prediction
data=data.replace(' ', np.nan)
data=data.dropna()

prob_missing = 0.1
data_incomplete = data.copy()
ix = [(row, col) for row in range(data.shape[0]) for col in range(data.shape[1])]
for row, col in random.sample(ix, int(round(prob_missing * len(ix)))):
    data_incomplete.iat[row, col] = np.nan
missing_encoded = pd.get_dummies(data_incomplete)
complete_encoded= pd.get_dummies(data)
print(complete_encoded.info(verbose=True))
for col in data.columns:
    missing_cols = missing_encoded.columns.str.startswith(str(col) + "_")
    missing_encoded.loc[data_incomplete[col].isnull(), missing_cols] = np.nan
print(missing_encoded.shape)
print(missing_encoded.values)
print(complete_encoded.shape)
print(complete_encoded.values.astype('float'))

#Change the dataFrame to the numpy array


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.xdim=20
        self.hdim=10
        self.zdim=5
        self.encode1 = nn.Linear(self.xdim, self.hdim)
        self.encodeMU = nn.Linear(self.hdim, self.zdim)
        self.encodelogvar = nn.Linear(self.hdim, self.zdim)
        
        self.decode1 = nn.Linear(self.zdim, self.hdim)
        self.decode2 = nn.Linear(self.hdim, self.xdim)
    
    def encode(self, x):
        h1 = F.relu(self.encode1(x))
        return self.encodeMU(h1), self.encodelogvar(h1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z):
        h = F.relu(self.decode1(z))
        rx = self.decode2(h)
        return rx.sigmoid()
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 20))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def createMask(data): #The function is used to create mask for the missing data
    miss_data=data.copy()
    missing_mask= np.isnan(data)  #bool matrix of mask
    miss_data[missing_mask] = -1.0  # fill data with -1
    return miss_data, missing_mask

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KLD

train_data=missing_encoded.values
complete_data=complete_encoded.values.astype('float')

device='cpu'
model = Autoencoder().to("cpu")
optimizer = optim.Adam(model.parameters(), lr=1e-3)
batch_size=64

train_num = len(train_data)

def next_batch(batch_size):
    idx=np.random.choice(train_num,batch_size)
    return train_data[idx],complete_data[idx]
total_loss=[]
error=[]
def train(epoch):
    num_epoch=epoch
    model.train()
    
    for e in range(num_epoch):
        train_loss = 0
        train_loss = 0
        for s in range(int(train_num/batch_size+1)):
            x,y=next_batch(batch_size)
            batch_data,mask=createMask(x) #fill the missing values with -1
            batch_data=torch.from_numpy(batch_data).float()
            y=torch.from_numpy(y).float()
            recon_batch, mu, logvar = model(batch_data)
            loss = loss_function(recon_batch, y, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        total_loss.append(train_loss/train_num)
        epoch_data,mask=createMask(train_data)
        actual = complete_data[mask]
        epoch_data=torch.from_numpy(epoch_data).float()
        recon,_,_=model(epoch_data)
        recon = recon.cpu().detach().numpy()
        index0=recon<0.5
        index1=recon>=0.5
        recon[index0]=0.0  #all the values less than 0.5 will be set as 0
        recon[index1]=1.0  #all the values larger than the 0.5 will be set as 1
        filled = recon[mask]  #use the mask to filter the fiiled entries
        error.append(np.count_nonzero(filled-actual))
        print(np.count_nonzero(filled-actual))  #count the number of wrong predictions
        print("\n epoch:{} ; loss:{}".format(e+1,train_loss/train_num))


train(100)
xAxis=[i for i in range(len(total_loss))]
plt.plot(xAxis, total_loss,label='total loss')
plt.legend()
plt.show()
xAxis=[i for i in range(len(error))]
plt.plot(xAxis, error,label='error')
plt.legend()
plt.show()






