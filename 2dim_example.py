# -*- coding: utf-8 -*-
#Example 1
#epsi = 0.01,sig = 0.2, sigma1=1,sigma2=2,xy=5,4
epsi = 0.01
sig = 0.2
sigma1 = 1

import numpy as np
import pandas as pd

XT = 5
YT = 4

def GeneratingData(T, dt, n_samples, n_re):
    #initial distribution
    Y0 = np.random.uniform(-YT,YT,n_samples)
    Y0 = Y0.repeat(n_re)
    X0 = np.random.uniform(-XT,XT,n_samples)
    X0 = X0.repeat(n_re)
    
    t = np.arange(0, T, dt)
    NT = len(t)+1
    x0 = X0[:]
    y0 = Y0[:]
    N = len(x0)
    x = np.zeros((NT, N))
    y = np.zeros((NT, N))
    x[0, :] = x0.squeeze()
    y[0, :] = y0.squeeze()
    
    for i in range(NT-1):
        UUt = dt**(1/2) * np.random.randn(N)
        VVt = dt**(1/2) * np.random.randn(N)
        x[i+1, :] = x[i, :] + (1*x[i, :] - x[i,:]*y[i, :])*dt + 0*x[i, :]*UUt+ sigma1*UUt
        y[i+1, :] = y[i, :] + (-1/epsi*y[i, :] + 1/(4*epsi)*x[i, :]**2)*dt + sig*epsi**(-1/2)*VVt    
    
    return x, y


import torch
import torch as th
import torch.nn as nn

data_size=(1200,2)    
num_epochs = 6000
learning_rate = 0.1
basis_num = 6 #1,x,y,x2,xy,y2

# +
class Y_drift(nn.Module):
    def __init__(self, basis_dim =6,fxy_dim =1):  #1,x,y,x2,xy,y2
        super().__init__()
        self.weights = nn.Linear(basis_dim ,fxy_dim ,bias=False)
    
    def forward(self, x):
        uf = self.weights(x)
        return uf

class Y_diff1(nn.Module):
    def __init__(self, basis_dim =1,sig_dim =1):  #1
        super().__init__()
        self.weights = nn.Linear(basis_dim ,sig_dim ,bias=False)
    
    def forward(self, x):
        sig = self.weights(x)
        return sig

class X_drift(nn.Module):
    def __init__(self, basis_dim =6,fxy_dim =1):  #1,x,y,x2,xy,y2
        super().__init__()
        self.weights = nn.Linear(basis_dim ,fxy_dim ,bias=False)
    
    def forward(self, x):
        uf = self.weights(x)
        return uf

class X_diff1(nn.Module):
    def __init__(self, basis_dim =1,sig_dim =1):  #1
        super().__init__()
        self.weights = nn.Linear(basis_dim ,sig_dim ,bias=False)
    
    def forward(self, x):
        sig = self.weights(x)
        return sig

# +
# weights=theta 
# Estimate slow variable weights
T = 0.0002
dt = 0.00002
n_samples = 1200
n_re = 200
#generate data
position_x, position_y = GeneratingData(T, dt, n_samples, n_re)
print("generate %d samples"% (n_samples))

posi_xt0 = position_x[0].reshape(-1,1)
posi_yt0 = position_y[0].reshape(-1,1)
position_xyt0 = np.concatenate((posi_xt0,posi_yt0),axis=-1)
    
posi_xt1 = position_x[-1].reshape(-1,1)
posi_yt1 = position_y[-1].reshape(-1,1)
position_xyt1 = np.concatenate((posi_xt1,posi_yt1),axis=-1)
    
position_xy = np.concatenate((position_xyt0,position_xyt1),axis=-1)
dxy01 = pd.DataFrame(position_xy,columns=['x0', 'y0' , 'x1', 'y1'])

# +
xy01 = dxy01.values

x0 = xy01[:,0].reshape(-1,n_re).mean(-1)
y0 = xy01[:,1].reshape(-1,n_re).mean(-1)
x1 = xy01[:,2].reshape(-1,n_re).mean(-1)
y1 = xy01[:,3].reshape(-1,n_re).mean(-1)

x1sq = ((xy01[:,2])**2).reshape(-1,n_re).mean(-1)
y1sq = ((xy01[:,3])**2).reshape(-1,n_re).mean(-1)

in1 = x0.reshape(-1,1)
in2 = y0.reshape(-1,1)
in3 = (x0**2).reshape(-1,1)
in4 = (y0*x0).reshape(-1,1)
in5 = (y0**2).reshape(-1,1)
in0 = np.ones(in1.shape)
indata = np.concatenate((in0,in1,in2,in3,in4,in5),axis=-1)
# -

xdtmean = (x1-x0)/0.0002
xdfmean = (x1sq+x0**2-2*x0*x1)/0.0002

# +
modeldrift = X_drift()
modeldiff = X_diff1()
optimizerdrift = th.optim.Adam(modeldrift.parameters(), lr=0.05)
optimizerdiff = th.optim.Adam(modeldiff.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(num_epochs):   
    print('epoch {}'.format(epoch + 1))
    print('*' * 10)
    reconst_loss = 0.0
    regular_loss = 0.0
    
    optimizerdrift.zero_grad()
    optimizerdiff.zero_grad()
    
       
    uf = modeldrift(th.Tensor(indata)).reshape(-1)
    lossfxy = criterion(uf,th.Tensor(xdtmean))
    usig = modeldiff(th.Tensor(indata[:,0:1])).reshape(-1)
    #usig = modeldiff(indata).reshape(-1)
    losssig = criterion(usig**2,th.Tensor(xdfmean))
    loss = lossfxy+losssig
    
    loss.backward() 
    nn.utils.clip_grad_norm_(modeldrift.parameters(), 100.)
    nn.utils.clip_grad_norm_(modeldiff.parameters(), 100.)
    optimizerdrift.step() 
    optimizerdiff.step() 
    reconst_loss = loss.item()
    
    if epoch >3000:
        for name, p in modeldrift.named_parameters():
            if name == 'weights.weight':
                for ii in range(6):
                    if torch.abs(p[0][ii]) < 0.5:
                        with torch.no_grad():
                            p[0][ii] = 0 
                            
# -

weidd=th.zeros([2,6])
for param_tensor in modeldrift.state_dict():
    weidd[0,:] = modeldrift.state_dict()[param_tensor]
for param_tensor in modeldiff.state_dict():
    weidd[1,0] = abs(modeldiff.state_dict()[param_tensor]) 
    #weidd[1,:] = modeldiff.state_dict()[param_tensor]  
#save data
df_weidd = pd.DataFrame(weidd.numpy())
#df_weidd.to_csv('data/EX1_xweight%d' %(10*sigma1),index=False)





# +
# Estimate fast variable weights
T = 0.00005
dt = 0.000005
n_samples = 1200
n_re = 50
#generate data
position_x, position_y = GeneratingData(T, dt, n_samples, n_re)
#print("generate %d samples"% (n_samples))

posi_xt0 = position_x[0].reshape(-1,1)
posi_yt0 = position_y[0].reshape(-1,1)
position_xyt0 = np.concatenate((posi_xt0,posi_yt0),axis=-1)
    
posi_xt1 = position_x[-1].reshape(-1,1)
posi_yt1 = position_y[-1].reshape(-1,1)
position_xyt1 = np.concatenate((posi_xt1,posi_yt1),axis=-1)
    
position_xy = np.concatenate((position_xyt0,position_xyt1),axis=-1)
dxy01 = pd.DataFrame(position_xy,columns=['x0', 'y0' , 'x1', 'y1'])

# +
xy01 = dxy01.values
x0 = xy01[:,0].reshape(-1,n_re).mean(-1)
y0 = xy01[:,1].reshape(-1,n_re).mean(-1)
x1 = xy01[:,2].reshape(-1,n_re).mean(-1)
y1 = xy01[:,3].reshape(-1,n_re).mean(-1)

x1sq = ((xy01[:,2])**2).reshape(-1,n_re).mean(-1)
y1sq = ((xy01[:,3])**2).reshape(-1,n_re).mean(-1)

in1 = x0.reshape(-1,1)
in2 = y0.reshape(-1,1)
in3 = (x0**2).reshape(-1,1)
in4 = (y0*x0).reshape(-1,1)
in5 = (y0**2).reshape(-1,1)
in0 = np.ones(in1.shape)
indata = np.concatenate((in0,in1,in2,in3,in4,in5),axis=-1)
# -

ydtmean = (y1-y0)/0.00005
ydfmean = (y1sq+y0**2-2*y0*y1)/0.00005


#training aepre
modeldrift = Y_drift()
modeldiff = Y_diff1()
optimizerdrift = th.optim.Adam(modeldrift.parameters(), lr=learning_rate)
optimizerdiff = th.optim.Adam(modeldiff.parameters(), lr=1)
criterion = nn.MSELoss()

for epoch in range(num_epochs):   # 训练所有!整套!数据次数
    print('epoch {}'.format(epoch + 1))
    print('*' * 10)
    reconst_loss = 0.0
    regular_loss = 0.0
    
    optimizerdrift.zero_grad()
    optimizerdiff.zero_grad()
    
       
    uf = modeldrift(th.Tensor(indata)).reshape(-1)
    lossfxy = criterion(uf,th.Tensor(ydtmean))
    usig = modeldiff(th.Tensor(indata[:,0:1])).reshape(-1)
    #usig = modeldiff(indata).reshape(-1)
    losssig = criterion(usig**2,th.Tensor(ydfmean))
    loss = lossfxy+losssig
    
    loss.backward() 
    nn.utils.clip_grad_norm_(modeldrift.parameters(), 100.)
    nn.utils.clip_grad_norm_(modeldiff.parameters(), 100.)
    optimizerdrift.step() 
    optimizerdiff.step() 
    reconst_loss = loss.item()
    
    if epoch >3000:
        for name, p in modeldrift.named_parameters():
            if name == 'weights.weight':
                for ii in range(6):
                    if torch.abs(p[0][ii]) < 2:
                        with torch.no_grad():
                            p[0][ii] = 0 


weidd=th.zeros([2,6])
for param_tensor in modeldrift.state_dict():
    weidd[0,:] = modeldrift.state_dict()[param_tensor]
for param_tensor in modeldiff.state_dict():
    weidd[1,0] = abs(modeldiff.state_dict()[param_tensor]) 
    #weidd[1,:] = modeldiff.state_dict()[param_tensor]  
#save data
df_weidd = pd.DataFrame(weidd.numpy())
#df_weidd.to_csv('data/EX1_yweight%d' %(10*sigma1),index=False)

#ae_pre
import torch
import torch as th
import torch.nn.functional as F
import torch.nn as nn
import h5py

# +
#1200*2,(20,200,600*2)
T = 0.02
dt = 0.001
n_samples = 1200
n_re = 1
position_x, position_y = GeneratingData(T, dt, n_samples, n_re)
Time = int(T/dt)+1
print("generate %d samples"% (n_samples))
position_xy = np.concatenate((np.split(position_x, n_samples*n_re, axis=1), np.split(position_y, n_samples*n_re, axis=1)), axis=1).reshape(-1,1200,Time).transpose(0,2,1)

#save data
# Create a new file
f = h5py.File('data/Ex1_xy_%d_samples_%d.h5'% (n_samples*n_re,10*sigma1), 'w')
f.create_dataset('dataset', data=position_xy)
f.close()
# -

ec_out_num_units = 50
batch_size = 2
num_epochs = 800
learning_rate = 0.001


class AE(nn.Module):
    def __init__(self, x_dim =1200, h1_dim =600, h2_dim =300, ec_out_num_units = 50):
        super(AE, self).__init__()
        self.en1 = nn.Linear(x_dim, h1_dim)
        self.en2 = nn.Linear(h1_dim, h2_dim)
        self.en3 = nn.Linear(h2_dim, ec_out_num_units)
        
        self.lstm = nn.LSTM(input_size=ec_out_num_units, hidden_size=ec_out_num_units, num_layers=1, bias=True, batch_first=True)
        
        self.de1 = nn.Linear(ec_out_num_units, h2_dim)
        self.de2 = nn.Linear(h2_dim, h1_dim)
        self.de3 = nn.Linear(h1_dim, x_dim)
        
    def encode(self, x):
        x = F.relu(self.en1(x))
        x = F.relu(self.en2(x))
        x = self.en3(x)
        return x
    
    def predict(self, x):
        pre_input = x[:,-2:,:]
        pre_fnt = x[:,2:,:]
        pre_bak, _ = self.lstm(pre_input)
        pre = th.cat([pre_fnt,pre_bak],dim = -2)
        return pre
 
    def decode(self, x):
        x = F.relu(self.de1(x))
        x = F.relu(self.de2(x))
        x = self.de3(x)
        return x
    
    def forward(self, x):
        x = self.encode(x)
        x = self.predict(x)
        aepredict = self.decode(x)
        return aepredict


#read data
df_xweidd = pd.read_csv('./data/EX1_xweight%d' %(10*sigma1))
xweidd = torch.Tensor(df_xweidd.values)
xdrift = xweidd[0,:]
xdiff = xweidd[1,:]

df_yweidd = pd.read_csv('./data/EX1_yweight%d' %(10*sigma1))
yweidd = torch.Tensor(df_yweidd.values)
ydrift = yweidd[0,:]
ydiff = yweidd[1,:]

def aepre_loss(aepre_out,indata):
    criterion = nn.MSELoss()
    
    out1 = aepre_out[:,:-2,:]
    tgt1 = indata[:,2:,:]
    out2 = aepre_out[:,-2:,:]
    eqin = indata[:,-2:,:].reshape(-1,2)
    tgt2x = eqin[:,0:1]
    tgt2y = eqin[:,1:2]
    UUt = 0.002**(1/2) * th.randn(1)
    VVt = 0.002**(1/2) * th.randn(tgt2y.shape)
    
    bas0 = th.ones(tgt2x.shape)
    bas1 = tgt2x
    bas2 = tgt2y
    bas3 = tgt2x**2
    bas4 = tgt2x*tgt2y
    bas5 = tgt2y**2
    basfuc = th.stack([bas0,bas1,bas2,bas3,bas4,bas5],0)
    
    tgt2xdri = th.zeros(tgt2x.shape)
    tgt2xdif = th.zeros(tgt2x.shape)
    for ii in range(6):
        tgt2xdri += xdrift[ii]*basfuc[ii]
        tgt2xdif += xdiff[ii]*basfuc[ii]
        
    tgt2ydri = th.zeros(tgt2y.shape)
    tgt2ydif = th.zeros(tgt2y.shape)
    for ii in range(6):
        tgt2ydri += ydrift[ii]*basfuc[ii]
        tgt2ydif += ydiff[ii]*basfuc[ii]
        
    tgt2xx = tgt2x + tgt2xdri*0.002 + tgt2xdif*UUt
    tgt2yy = tgt2y + tgt2ydri*0.002 + tgt2ydif*VVt
    tgt2 = th.cat([tgt2xx,tgt2yy],dim = -1).reshape(out2.shape)
    aepre_loss = criterion(out1,tgt1)+criterion(out2,tgt2)
    return aepre_loss

# read data 
with h5py.File('./data/Ex1_xy_%d_samples_%d.h5'% (n_samples,10*sigma1), "r") as hf:
    dataset = hf['dataset'][:]
hf.close()

dataset = torch.Tensor(dataset)
dataset_ae = dataset[:,1:11,:]

import torch.utils.data as Data
train_dataset = Data.TensorDataset(dataset_ae)  
data_loader = Data.DataLoader(
    dataset=train_dataset,      # torch TensorDataset format
    batch_size=batch_size,           # mini batch size 
    shuffle=True,          
)

#training
modelaepre = AE()
optimizer = th.optim.Adam(modelaepre.parameters(), lr=learning_rate)

training_loss = []  #training_loss.append(reconst_loss)
#0-10~2-12
for epoch in range(num_epochs):   
    print('epoch {}'.format(epoch + 1))
    print('*' * 10)
    reconst_loss = 0.0
    aveloss = []
    for step, batch_xx in enumerate(data_loader): 
        optimizer.zero_grad()
        
        in_data = torch.Tensor([item.detach().numpy() for item in batch_xx]).reshape(-1,10,1200)
        aepredict = modelaepre(in_data)
        loss = aepre_loss(aepredict, in_data)
        
        loss.backward() 
        nn.utils.clip_grad_norm_(modelaepre.parameters(), 100.)
        optimizer.step() 
        reconst_loss = loss.item()
        aveloss.append(reconst_loss)
        
        #print( epoch, step, batch_xx)
        if (step+1) % 2 == 0:
            print ("Epoch[{}/{}], Step [{}/{}], Loss: {:.4f}" 
                   .format(epoch+1, num_epochs, step+1, len(data_loader), reconst_loss))
    
    if (epoch+1) % 100 == 0:
        aver = sum(aveloss)/len(aveloss)
        training_loss.append(aver)
        aveloss = []

aepredictl = modelaepre(dataset_ae.to(torch.float32))

#2-12~10-20
for ii in range(4):
    train_dataset1 = Data.TensorDataset(aepredictl)  
    data_loader1 = Data.DataLoader(
        dataset=train_dataset1,      # torch TensorDataset format
        batch_size=batch_size,           # mini batch size 
        shuffle=True,          # shuffle on the batch
    )
    for epoch in range(num_epochs):   
        print('epoch {}'.format(epoch + 1))
        print('*' * 10)
        reconst_loss = 0.0
        aveloss = []
        for step, batch_xx in enumerate(data_loader1):  
            
            optimizer.zero_grad()
            
            in_data = torch.Tensor([item.detach().numpy() for item in batch_xx]).reshape(-1,10,1200)
            aepredict = modelaepre(in_data)
            
            loss = aepre_loss(aepredict, in_data)
            
            loss.backward() 
            nn.utils.clip_grad_norm_(modelaepre.parameters(), 100.)
            optimizer.step() 
            reconst_loss = loss.item()
            aveloss.append(reconst_loss)
            
            #print( epoch, step, batch_xx)
            if (step+1) % 2 == 0:
                print ("Epoch[{}/{}], Step [{}/{}], Loss: {:.4f}" 
                       .format(epoch+1, num_epochs, step+1, len(data_loader), reconst_loss))
        if (epoch+1) % 100 == 0:
            aver = sum(aveloss)/len(aveloss)
            training_loss.append(aver)
            aveloss = []
               
    aepredictl = modelaepre(aepredictl.to(torch.float32))

#save pre_20
pre_20_save = aepredictl[:,:,:].detach().numpy()
fff = h5py.File('./data/Ex1_pre_20_%d.h5'% (10*sigma1),'w')
fff.create_dataset('pre_20', data=pre_20_save)
fff.close()


#10-20~30-40
for ii in range(5):
    train_dataset1 = Data.TensorDataset(aepredictl)  
    data_loader1 = Data.DataLoader(
        dataset=train_dataset1,      # torch TensorDataset format
        batch_size=batch_size,           # mini batch size 
        shuffle=True,          # shuffle on the batch
    )
    for epoch in range(num_epochs):   
        print('epoch {}'.format(epoch + 1))
        print('*' * 10)
        reconst_loss = 0.0
        aveloss = []
        for step, batch_xx in enumerate(data_loader1):  
            
            optimizer.zero_grad()
            
            in_data = torch.Tensor([item.detach().numpy() for item in batch_xx]).reshape(-1,10,1200)
            aepredict = modelaepre(in_data)
            
            loss = aepre_loss(aepredict, in_data)
            
            loss.backward() 
            nn.utils.clip_grad_norm_(modelaepre.parameters(), 100.)
            optimizer.step() 
            reconst_loss = loss.item()
            aveloss.append(reconst_loss)
            
            #print( epoch, step, batch_xx)
            if (step+1) % 2 == 0:
                print ("Epoch[{}/{}], Step [{}/{}], Loss: {:.4f}" 
                       .format(epoch+1, num_epochs, step+1, len(data_loader), reconst_loss))
        if (epoch+1) % 100 == 0:
            aver = sum(aveloss)/len(aveloss)
            training_loss.append(aver)
            aveloss = []
            
    aepredictl = modelaepre(aepredictl.to(torch.float32))

#save pre_30
pre_30_save = aepredictl[:,:,:].detach().numpy()

fff = h5py.File('./data/Ex1_pre_30_%d.h5'% (10*sigma1),'w')
fff.create_dataset('pre_30', data=pre_30_save)
fff.close()

for ii in range(5):
    train_dataset1 = Data.TensorDataset(aepredictl)  
    data_loader1 = Data.DataLoader(
        dataset=train_dataset1,      # torch TensorDataset format
        batch_size=batch_size,           # mini batch size 
        shuffle=True,          # shuffle on the batch
    )
    for epoch in range(num_epochs):   
        print('epoch {}'.format(epoch + 1))
        print('*' * 10)
        reconst_loss = 0.0
        aveloss = []
        for step, batch_xx in enumerate(data_loader1):  
            
            optimizer.zero_grad()
            
            in_data = torch.Tensor([item.detach().numpy() for item in batch_xx]).reshape(-1,10,1200)
            aepredict = modelaepre(in_data)
            
            loss = aepre_loss(aepredict, in_data)
            
            loss.backward() 
            nn.utils.clip_grad_norm_(modelaepre.parameters(), 100.)
            optimizer.step() 
            reconst_loss = loss.item()
            aveloss.append(reconst_loss)
            
            #print( epoch, step, batch_xx)
            if (step+1) % 2 == 0:
                print ("Epoch[{}/{}], Step [{}/{}], Loss: {:.4f}" 
                       .format(epoch+1, num_epochs, step+1, len(data_loader), reconst_loss))
        if (epoch+1) % 100 == 0:
            aver = sum(aveloss)/len(aveloss)
            training_loss.append(aver)
            aveloss = []
            
    aepredictl = modelaepre(aepredictl.to(torch.float32))

#save pre_40
pre_40_save = aepredictl[:,:,:].detach().numpy()

fff = h5py.File('./data/Ex1_pre_40_%d.h5'% (10*sigma1),'w')
fff.create_dataset('pre_40', data=pre_40_save)
fff.close()

#30-40~40-50
for ii in range(5):
    train_dataset1 = Data.TensorDataset(aepredictl)  
    data_loader1 = Data.DataLoader(
        dataset=train_dataset1,      # torch TensorDataset format
        batch_size=batch_size,           # mini batch size 
        shuffle=True,          # shuffle on the batch
    )
    for epoch in range(num_epochs):   
        print('epoch {}'.format(epoch + 1))
        print('*' * 10)
        reconst_loss = 0.0
        aveloss = []
        for step, batch_xx in enumerate(data_loader1):  
            
            optimizer.zero_grad()
            
            in_data = torch.Tensor([item.detach().numpy() for item in batch_xx]).reshape(-1,10,1200)
            aepredict = modelaepre(in_data)
            
            loss = aepre_loss(aepredict, in_data)
            
            loss.backward() 
            nn.utils.clip_grad_norm_(modelaepre.parameters(), 100.)
            optimizer.step() 
            reconst_loss = loss.item()
            aveloss.append(reconst_loss)
            
            #print( epoch, step, batch_xx)
            if (step+1) % 2 == 0:
                print ("Epoch[{}/{}], Step [{}/{}], Loss: {:.4f}" 
                       .format(epoch+1, num_epochs, step+1, len(data_loader), reconst_loss))
        if (epoch+1) % 100 == 0:
            aver = sum(aveloss)/len(aveloss)
            training_loss.append(aver)
            aveloss = []
            
    aepredictl = modelaepre(aepredictl.to(torch.float32))

#save pre_50
pre_50_save = aepredictl[:,:,:].detach().numpy()
fff = h5py.File('./data/Ex1_pre_50_%d.h5'% (10*sigma1),'w')
fff.create_dataset('pre_50', data=pre_50_save)
fff.close()


# +
df_loss = pd.DataFrame(np.array(training_loss))
df_loss.to_csv('data/Ex1_loss%d' %(10*sigma1),index=False)

"""
#60
for ii in range(5):
    train_dataset1 = Data.TensorDataset(aepredictl)  
    data_loader1 = Data.DataLoader(
        dataset=train_dataset1,      # torch TensorDataset format
        batch_size=batch_size,           # mini batch size 
        shuffle=True,          # shuffle on the batch
    )
    for epoch in range(num_epochs):   
        print('epoch {}'.format(epoch + 1))
        print('*' * 10)
        reconst_loss = 0.0
        aveloss = []
        for step, batch_xx in enumerate(data_loader1):  
            
            optimizer.zero_grad()
            
            in_data = torch.Tensor([item.detach().numpy() for item in batch_xx]).reshape(-1,10,1200)
            aepredict = modelaepre(in_data)
            
            loss = aepre_loss(aepredict, in_data)
            
            loss.backward() 
            nn.utils.clip_grad_norm_(modelaepre.parameters(), 100.)
            optimizer.step() 
            reconst_loss = loss.item()
            aveloss.append(reconst_loss)
            
            #print( epoch, step, batch_xx)
            if (step+1) % 2 == 0:
                print ("Epoch[{}/{}], Step [{}/{}], Loss: {:.4f}" 
                       .format(epoch+1, num_epochs, step+1, len(data_loader), reconst_loss))
        if (epoch+1) % 100 == 0:
            aver = sum(aveloss)/len(aveloss)
            training_loss.append(aver)
            aveloss = []
            
    aepredictl = modelaepre(aepredictl.to(torch.float32))

#save pre_60
pre_60_save = aepredictl[:,:,:].detach().numpy()
fff = h5py.File('./data/SVDP_pre_60_%d.h5'% (10*sigma1),'w')
fff.create_dataset('pre_60', data=pre_60_save)
fff.close()

#70
for ii in range(5):
    train_dataset1 = Data.TensorDataset(aepredictl)  
    data_loader1 = Data.DataLoader(
        dataset=train_dataset1,      # torch TensorDataset format
        batch_size=batch_size,           # mini batch size 
        shuffle=True,          # shuffle on the batch
    )
    for epoch in range(num_epochs):   
        print('epoch {}'.format(epoch + 1))
        print('*' * 10)
        reconst_loss = 0.0
        for step, batch_xx in enumerate(data_loader1):  
            
            optimizer.zero_grad()
            
            in_data = torch.Tensor([item.detach().numpy() for item in batch_xx]).reshape(-1,10,1200)
            aepredict = modelaepre(in_data)
            
            loss = aepre_loss(aepredict, in_data)
            
            loss.backward() 
            nn.utils.clip_grad_norm_(modelaepre.parameters(), 100.)
            optimizer.step() 
            reconst_loss = loss.item()
            
            #print( epoch, step, batch_xx)
            if (step+1) % 2 == 0:
                print ("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}" 
                       .format(epoch+1, num_epochs, step+1, len(data_loader), reconst_loss))
            
    aepredictl = modelaepre(aepredictl.to(torch.float32))

#save pre_70
pre_70_save = aepredictl[:,:,:].detach().numpy()
fff = h5py.File('./data/SVDP_pre_70_%d.h5'% (10*sigma1),'w')
fff.create_dataset('pre_70', data=pre_70_save)
fff.close()

#80
for ii in range(5):
    train_dataset1 = Data.TensorDataset(aepredictl)  
    data_loader1 = Data.DataLoader(
        dataset=train_dataset1,      # torch TensorDataset format
        batch_size=batch_size,           # mini batch size 
        shuffle=True,          # shuffle on the batch
    )
    for epoch in range(num_epochs):   
        print('epoch {}'.format(epoch + 1))
        print('*' * 10)
        reconst_loss = 0.0
        for step, batch_xx in enumerate(data_loader1):  
            
            optimizer.zero_grad()
            
            in_data = torch.Tensor([item.detach().numpy() for item in batch_xx]).reshape(-1,10,1200)
            aepredict = modelaepre(in_data)
            
            loss = aepre_loss(aepredict, in_data)
            
            loss.backward() 
            nn.utils.clip_grad_norm_(modelaepre.parameters(), 100.)
            optimizer.step() 
            reconst_loss = loss.item()
            
            #print( epoch, step, batch_xx)
            if (step+1) % 2 == 0:
                print ("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}" 
                       .format(epoch+1, num_epochs, step+1, len(data_loader), reconst_loss))
            
    aepredictl = modelaepre(aepredictl.to(torch.float32))

#save pre_80
pre_80_save = aepredictl[:,:,:].detach().numpy()
fff = h5py.File('./data/SVDP_pre_80_%d.h5'% (10*sigma1),'w')
fff.create_dataset('pre_80', data=pre_80_save)
fff.close()
"""
# -


