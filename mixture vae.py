
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
data=pd.read_csv('Tab.delimited.Cleaned.dataset.WITH.variable.labels.csv', sep='\t', engine='python')
file=open('mixture.txt')
labels=[]
for line in file:
    word=line.rstrip('\n')
    labels.append(word)

data=data.loc[:,labels]  #The selected columns prediction
data=data.replace(' ', np.nan)
#print(data.info(verbose=True))
data=data.dropna()
#print(data.info(verbose=True))
#17 rows and 49 cols
file=open('mixture.txt')
categories=["diseaseframinga","reciprocityothera","reciprocityusa","allowedforbiddena","flagtimeestimate1",
            "flagtimeestimate2","flagtimeestimate3","flagtimeestimate4","sex","citizenship"]
df=pd.get_dummies(data,columns=categories)
#print(df.info(verbose=True))

df=df.replace("Very much","11")
#rint(train.loc[:,"flagsupplement1"])
df=df.replace("Not at all","1")
df=df.replace("Republican","7")
df=df.replace("Democrat","1")
df=df.replace("Conservative","7")
df=df.replace("Liberal","1")
df=df.replace("Moderately Disgusting", '6')
df=df.replace("Moderately bad", '6')
df=df.replace("Very Avoid", '7')
df=df.replace("Slightly bad", '5')
df=df.replace("Moderately Sad", '6')
df=df.replace("Very Afraid", '7')
df=df.replace("Very Sad", '7')
df=df.replace("Slightly Disgusting", '5')
df=df.replace("Moderately Avoid", '6')
df=df.replace("Moderately Afraid", '6')
df=df.replace("Moderately Ugly", '6')
df=df.replace("Very Disgusting", '7')
df=df.replace("Slightly Afraid", '5')
df=df.replace("Very bad", '7')
df=df.replace("Very Ugly", '7')
df=df.replace("Slightly Sad", '5')
df=df.replace("Slightly Ugly", '5')
df=df.replace("Slightly Avoid", '5')
df=df.replace("Strongly disagree", '1')
df=df.replace("Strongly agree", '7')
df= df[:].astype(float)
print(df.info(verbose=True))
#The above part is preprocessing the data and represent it with numpy array
prob_missing = 0.1
data_incomplete = df.copy()
ix = [(row, col) for row in range(df.shape[0]) for col in range(df.shape[1])]
for row, col in random.sample(ix, int(round(prob_missing * len(ix)))):
    data_incomplete.iat[row, col] = -1.0
#
#print(data_incomplete)
complete_data=df.values.astype('float')
train_data=data_incomplete.values.astype('float')
#print(train_data.shape)  #The data has 42 rows and 68 cols

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.xdim=68
        self.hdim=20
        self.zdim=10
        self.BN1= nn.BatchNorm1d(20)
        self.encode1 = nn.Linear(self.xdim, self.hdim)
        self.encodeMU = nn.Linear(self.hdim, self.zdim)
        self.encodelogvar = nn.Linear(self.hdim, self.zdim)
        
        self.decode1 = nn.Linear(self.zdim, self.hdim)
        self.decode2 = nn.Linear(self.hdim, self.xdim)
    
    def encode(self, x):
        h1 = F.relu(self.encode1(x))
        h1 = self.BN1(h1)
        return self.encodeMU(h1), self.encodelogvar(h1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z):
        h = F.relu(self.decode1(z))
        h = self.BN1(h)
        rx = self.decode2(h)
        y=rx[:,38:].sigmoid() #From 38 to 67 apply sigmoid function
        rx[:,38:]=y
        return rx
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 68))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
'''
    def createMask(data): #The function is used to create mask for the missing data
    miss_data=data.copy()
    missing_mask= np.isnan(data)  #bool matrix of mask
    miss_data[missing_mask] = -1.0  # fill data with -1
    return miss_data, missing_mask
    '''
def loss_function(recon_x, x, mu, logvar):
    recon_real=recon_x[:,:38]   #cols 0-37 are real values
    recon_cat=recon_x[:,38:]  #cols 38-67 are categorical values
    
    x_real=x[:,:38]
    x_cat=x[:,38:]
    
    BCE = F.binary_cross_entropy(recon_cat,x_cat, reduction='mean') #The loss
    
    loss= torch.nn.SmoothL1Loss(reduction='mean')
    MSE = loss(recon_real,x_real)
    
    
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    
    return (0.0001*MSE)+BCE+KLD, BCE, MSE


#train_data = torch.from_numpy(train_data).float()

batch_size=8  #48 rows

#print(complete_data[:,38:])

train_num = len(train_data)

def next_batch(batch_size):
    idx=np.random.choice(train_num,batch_size)
    return train_data[idx],complete_data[idx]
use_cuda=False
device = torch.device("cuda" if use_cuda else "cpu")
model = Autoencoder().to("cpu")
optimizer = optim.SGD(model.parameters(), lr=1e-05, momentum=0.9, weight_decay=5e-4)
num_epochs=100
huber_loss=[]
bce_loss=[]
total_loss=[]
for epoch in range(num_epochs):
    model.train()
    total_batch = len(train_data) // batch_size
    
    for i in range(int(total_batch/batch_size)+1):
        batch_data,actual=next_batch(batch_size)
        #print("batch data")
        
        batch_data=torch.from_numpy(batch_data).float()
        actual=torch.from_numpy(actual).float()
        
        batch_data = batch_data.to(device)
        #print(batch_data[:,38:])
        actual=actual.to(device)
        
        optimizer.zero_grad()
        
        recon_batch, mu, logvar = model(batch_data)
        #print(actual[:,38:])
        loss,bce,mse = loss_function(recon_batch, actual, mu, logvar)
        # Add the L1 regularization
        all_linear1_params = torch.cat([x.view(-1) for x in model.encode1.parameters()])
        l1_regularization = 0.5 * torch.norm(all_linear1_params, 1)
        loss=loss+l1_regularization
        
        total_loss.append(loss.item())
        huber_loss.append(mse.item())
        bce_loss.append(bce.item())
        
        loss.backward()
        
        optimizer.step()
        
        if (i) ==(total_batch//2):
            print('Epoch [%d/%d], lter [%d/%d], Loss: %.6f'
                  %(epoch+1, num_epochs, i+1, total_batch, loss.item()))


print(model.encode1)

xAxis=[i for i in range(len(total_loss))]
plt.plot(xAxis, total_loss,label='total loss')
plt.legend()
plt.plot(xAxis, bce_loss,label='bce loss')

plt.legend()
plt.plot(xAxis, huber_loss,label='mse loss')
plt.legend()
plt.show()
print(bce_loss)
