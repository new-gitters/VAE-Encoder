import torch
from torch import nn, optim
import torch.autograd as autograd
import torch.utils.data
import numpy as np
from torch.nn import functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import pandas as pd
from torch.autograd import Variable
from math import sqrt
from sklearn.metrics import mean_squared_error

data=pd.read_csv('Tab.delimited.Cleaned.dataset.WITH.variable.labels.csv', sep='\t', engine='python')

#print(data.loc[:,"IATEXPfilter"])
file=open('ordinal.txt')
labels=[]
for line in file:
    word=line.rstrip('\n')   
    labels.append(word)
selected=data.loc[:,labels]  #The selected columns prediction
#print(selected.info(verbose=True))
selected=selected.replace(' ', np.nan)
print(selected.info(verbose=True))
train=selected.dropna()
print(train.info(verbose=True))
#train.loc[:, ["flagsupplement1","flagsupplement2","flagsupplement3"]].replace(["Very much","Not at all","Republican","Democrat","Conservative","Liberal"], ["11", "1","7","1","7","1"], inplace=True)

train=train.replace("Very much","11")
#rint(train.loc[:,"flagsupplement1"])
train=train.replace("Not at all","1")
train=train.replace("Republican","7")
train=train.replace("Democrat","1")
train=train.replace("Conservative","7")
train=train.replace("Liberal","1")
train=train.replace("Moderately Disgusting", '6')
train=train.replace("Moderately bad", '6')
train=train.replace("Very Avoid", '7')
train=train.replace("Slightly bad", '5')
train=train.replace("Moderately Sad", '6')
train=train.replace("Very Afraid", '7')
train=train.replace("Very Sad", '7')
train=train.replace("Slightly Disgusting", '5')
train=train.replace("Moderately Avoid", '6')
train=train.replace("Moderately Afraid", '6')
train=train.replace("Moderately Ugly", '6')
train=train.replace("Very Disgusting", '7')
train=train.replace("Slightly Afraid", '5')
train=train.replace("Very bad", '7')
train=train.replace("Very Ugly", '7')
train=train.replace("Slightly Sad", '5')
train=train.replace("Slightly Ugly", '5')
train=train.replace("Slightly Avoid", '5')
train=train.replace("Strongly disagree", '1')
train=train.replace("Strongly agree", '7')
training=train.to_dict(orient='records')

data=[]
label=[]
label=list(training[0].keys())
for item in training:
    temp=list(item.values())
    
    value=[int(x) for x in temp]
    data.append(value) #Transfer the diction to list type

#The shape of the data is (5698, 35)
#centralize and normalize the data
data=np.array(data,float)
data-=np.mean(data, axis=0) 
data/= np.std(data, axis=0)  
print(data.shape)
row,cols=data.shape
test_data=data.copy()
train_data=torch.from_numpy(data).float()

#The function is used to generate the missing data point and mask
def missing_method(raw_data,method='random') :
    
    data = raw_data.copy()
    rows, cols = data.shape
    
    # missingness threshold
    t = 0.2
    if method == 'uniform' :
            # uniform random vector
        v = np.random.uniform(size=(rows, cols))

            # missing values where v<=t
        mask = (v<=t)
        data[mask] = 0

    elif method == 'random' :
            # only half of the attributes to have missing value
        missing_cols = np.random.choice(cols, cols//2)
        c = np.zeros(cols, dtype=bool)
        c[missing_cols] = True

            # uniform random vector
        v = np.random.uniform(size=(rows, cols))

            # missing values where v<=t
        mask = (v<=t)*c
        data[mask] = 0

    else :
        print("Error : There are no such method")
        raise
    return data, mask
    
dropout_ratio = 0.2
    
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.drop_out = nn.Dropout(p=0.2)
        self.xdim=35
        self.hdim=15
        self.h2dim=10
        self.zdim=5
        self.encode1 = nn.Linear(self.xdim, self.hdim)
        self.encode2 = nn.Linear(self.hdim, self.h2dim)
        self.encodeMU = nn.Linear(self.h2dim, self.zdim)
        self.encodelogvar = nn.Linear(self.h2dim, self.zdim)
        
        self.decode1 = nn.Linear(self.zdim, self.h2dim)
        self.decode2 = nn.Linear(self.h2dim, self.hdim)
        self.decode3 = nn.Linear(self.hdim, self.xdim)

    def encode(self, x):
        h1 = F.relu(self.encode1(x))
        h2 = F.relu(self.encode2(h1))
        return self.encodeMU(h2), self.encodelogvar(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z):
        d1 = F.relu(self.decode1(z))
        d2 = F.relu(self.decode2(d1))
        return self.decode3(d2)
    
    def forward(self, x):
        x_missed = self.drop_out(x)
        mu, logvar = self.encode(x_missed.view(-1, 35))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

#device = torch.device("cuda" if args.cuda else "cpu")
device='cpu'
model = Autoencoder().to("cpu")
optimizer = optim.Adam(model.parameters(), lr=1e-3)
train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=64,
                                           shuffle=True)
def loss_function(recon_x, x, mu, logvar):
    loss = nn.MSELoss()
    MSE = loss(recon_x,x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + KLD
total_loss=[]
test_loss=[]
missed_data, mask = missing_method(test_data, method='random' )
missed_data = torch.from_numpy(missed_data).float()
def train(num_epochs):
    model.train()
    
    for epoch in range(num_epochs):
        train_loss = 0
        total_batch = len(train_data) // 64
        for batch_idx, batch_data in enumerate(train_loader):
            datbatch_data = batch_data.to(device)
            optimizer.zero_grad()
            batch_data=batch_data.view(-1, 35)
            recon_batch, mu, logvar = model(batch_data)
            loss = loss_function(recon_batch, batch_data, mu, logvar)
            
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(batch_data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(batch_data)))
        total_loss.append(train_loss)
        print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
        
        model.eval()
        filled_data, mu, logvar = model(missed_data)

        filled_data = filled_data.cpu().detach().numpy()

        rmse_sum = 0

        for i in range(cols) :
            if mask[:,i].sum() > 0 :
                y_actual = test_data[:,i][mask[:,i]]
                y_predicted = filled_data[:,i][mask[:,i]]

                rmse = np.sqrt(mean_squared_error(y_actual, y_predicted))
                rmse_sum += rmse
        test_loss.append(rmse_sum)
          

train(50)

xAxis=[i for i in range(len(total_loss))]
plt.plot(xAxis, total_loss,label='train loss')
plt.legend()
plt.show()
plt.plot(xAxis, test_loss,label='test loss')
plt.legend()
plt.show()

