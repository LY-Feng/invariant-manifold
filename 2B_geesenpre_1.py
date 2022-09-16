# -*- coding: utf-8 -*-
"""
Created on Sat

@author: DELL
"""
#epsi = 0.01,sig = 0.1, sigma1=1,sigma2=2,xy=5,6

    
epsi = 0.01
sig = 0.1
sigma1 = 1

import numpy as np
import h5py
XT = 5
YT = 6
def GeneratingData(T, dt, n_samples, n_re):
    #initial distribution
    Y0 = np.random.uniform(-YT,YT,n_samples)
    Y0 = np.tile(Y0, n_re)
    X0 = np.random.uniform(-XT,XT,n_samples)
    X0 = X0.repeat(n_re)
    np.random.shuffle(X0)
    np.random.shuffle(Y0)
    
    t = np.arange(0, T, dt)
    NT = len(t)
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
        y[i+1, :] = y[i, :] + (-1/epsi*y[i, :] + 1/(4*epsi)*x[i, :]**2)*dt + 0*y[i, :]*VVt + sig*epsi**(-1/2)*VVt    
    
    return x, y


if __name__ == "__main__":
    
    T = 0.01
    dt = 0.001
    n_samples = 200
    n_re = 100
    #generate data
    position_x, position_y = GeneratingData(T, dt, n_samples, n_re)
    print("generate %d samples"% (n_samples * n_re))
    Time = int(T/dt)
    #(10,1200*2)
    posi_xt0 = position_x[0:9].reshape(-1,1)
    posi_yt0 = position_y[0:9].reshape(-1,1)
    position_xyt0 = np.concatenate((posi_xt0,posi_yt0),axis=-1)
    
    posi_xt1 = position_x[1:10].reshape(-1,1)
    posi_yt1 = position_y[1:10].reshape(-1,1)
    position_xyt1 = np.concatenate((posi_xt1,posi_yt1),axis=-1)
    
    #save data
    # Create a new file
    f = h5py.File('data/2B_xy_%d_samples_nonesti%d.h5' %(n_samples* n_re,10*sigma1), 'w')
    f.create_dataset('xyt0', data=position_xyt0)
    f.create_dataset('xyt1', data=position_xyt1)
    f.close()  
 
    
    #1200*2,(20,200,600*2)
    T = 0.041
    dt = 0.001
    n_samples = 120
    n_re = 10
    Time = int(T/dt)
    position_x, position_y = GeneratingData(T, dt, n_samples, n_re)
    print("generate %d samples"% (n_samples * n_re))
    position_xy = np.concatenate((np.split(position_x, n_samples*n_re, axis=1), np.split(position_y, n_samples*n_re, axis=1)), axis=1).reshape(-1,1200,Time).transpose(0,2,1)
    position_xy_input = position_xy[:,:,:]
    
    #save data
    # Create a new file
    f = h5py.File('data/2BB_xy_%d_samples_enpredeesti%d.h5'% (n_samples*n_re,10*sigma1), 'w')
    f.create_dataset('dataset', data=position_xy_input)
    f.close()


#estimate weight
##weight=theta    
import torch
import torch as th
import torch.nn as nn
import h5py
import pandas as pd

data_size=(20000*9,2)    
num_epochs = 5000
learning_rate = 0.001
basis_num = 6 #1,x,y,x2,xy,y2

class X_drift(nn.Module):
    def __init__(self, basis_dim =6,fxy_dim =1):  #1,x,y,x2,xy,y2
        super(X_drift, self).__init__()
        self.weights = nn.Linear(basis_dim ,fxy_dim ,bias=False)
    
    def forward(self, x):
        uf = self.weights(x)
        return uf
    
class X_diff(nn.Module):
    def __init__(self, basis_dim =6,sig_dim =1):  #1,x,y,x2,xy,y2
        super(X_diff, self).__init__()
        self.weights = nn.Linear(basis_dim ,sig_dim ,bias=False)
    
    def forward(self, x):
        sig = self.weights(x)
        return sig
    
class X_diff1(nn.Module):
    def __init__(self, basis_dim =1,sig_dim =1):  #1
        super(X_diff1, self).__init__()
        self.weights = nn.Linear(basis_dim ,sig_dim ,bias=False)
    
    def forward(self, x):
        sig = self.weights(x)
        return sig
    

def esti_loss(uf,uxt0,ut1):
    criterion = nn.MSELoss()
    uxt1 = ut1[:,0]
    fxt0 = (uxt1-uxt0)/0.001
    u_loss = criterion(fxt0,uf)
    return u_loss

def sig_loss(sig1,uxt0,ut1):
    criterion = nn.MSELoss()
    uxt1 = ut1[:,0]
    sig0 = ((uxt1-uxt0)**2/0.001)
    u_loss = criterion(sig0,sig1**2)
    return u_loss

# read data 
with h5py.File('data/2B_xy_20000_samples_nonesti%d.h5' %(10*sigma1), "r") as hf:
    dataset0 = hf['xyt0'][:]
    dataset1 = hf['xyt1'][:]
hf.close()

dataset0 = torch.Tensor(dataset0)
dataset1 = torch.Tensor(dataset1)

uxt0 = dataset0[:,0]
uyt0 = dataset0[:,1]
uxt1 = dataset1[:,0]

in1 = uxt0.reshape(-1,1)
in2 = uyt0.reshape(-1,1)
in3 = (uxt0**2).reshape(-1,1)
in4 = (uyt0*uxt0).reshape(-1,1)
in5 = (uyt0**2).reshape(-1,1)
in0 = th.ones(in1.shape)
indata = th.cat((in0,in1,in2,in3,in4,in5),dim=-1)

#training aepre
modeldrift = X_drift()
modeldiff = X_diff1()
optimizerdrift = th.optim.Adam(modeldrift.parameters(), lr=learning_rate)
optimizerdiff = th.optim.Adam(modeldiff.parameters(), lr=learning_rate)

for epoch in range(num_epochs):   # 训练所有!整套!数据次数
    print('epoch {}'.format(epoch + 1))
    print('*' * 10)
    reconst_loss = 0.0
    regular_loss = 0.0
    
    optimizerdrift.zero_grad()
    optimizerdiff.zero_grad()
    
    
    for param in modeldrift.parameters():
        regular_loss += th.sum(th.abs(param[0][4]))+th.sum(th.abs(param[0][1]))
        break
        
    uf = modeldrift(indata).reshape(-1)
    lossfxy = esti_loss(uf,uxt0,dataset1)
    usig = modeldiff(in0).reshape(-1)
    #usig = modeldiff(indata).reshape(-1)
    losssig = sig_loss(usig,uxt0,dataset1)
    loss = lossfxy+losssig
    
    #反向传播计算梯度
    loss.backward() 
    nn.utils.clip_grad_norm_(modeldrift.parameters(), 100.)
    nn.utils.clip_grad_norm_(modeldiff.parameters(), 100.)
    optimizerdrift.step() #应用梯度
    optimizerdiff.step() 
    reconst_loss = loss.item()
    
    if epoch >2800:
        for name, p in modeldrift.named_parameters():
            if name == 'weights.weight':
                for ii in range(6):
                    if torch.abs(p[0][ii]) < 0.05:
                        with torch.no_grad():
                            p[0][ii] = 0 
        """                    
        for name, p in modeldiff.named_parameters():
            if name == 'weights.weight':
                for ii in range(6):
                    if torch.abs(p[0][ii]) < 0.01:
                        with torch.no_grad():
                            p[0][ii] = 0 
        """
    
weidd=th.zeros([2,6])
for param_tensor in modeldrift.state_dict():
    weidd[0,:] = modeldrift.state_dict()[param_tensor]
for param_tensor in modeldiff.state_dict():
    weidd[1,0] = abs(modeldiff.state_dict()[param_tensor]) 
    #weidd[1,:] = modeldiff.state_dict()[param_tensor]  
#save data
df_weidd = pd.DataFrame(weidd.numpy())
df_weidd.to_csv('data/2B_200000samples_xweight%d' %(10*sigma1),index=False)

#enpre
import torch
import torch as th
import torch.nn.functional as F
import torch.nn as nn
import numpy
import h5py
import pandas as pd

data_size=(10,1200*2)    #(2,200,1200) 
ec_out_num_units = 50
batch_size = 1
num_epochs = 200
learning_rate = 0.001


# 定义AE模型
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
df_xweidd = pd.read_csv('./data/2B_200000samples_xweight%d' %(10*sigma1))

xweidd = torch.Tensor(df_xweidd.values)
xdrift = xweidd[0,:]
xdiff = xweidd[1,:]
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
    tgt2xx = tgt2x + tgt2xdri*0.002 + tgt2xdif*UUt
    tgt2yy = tgt2y + (-1/epsi*tgt2y + 1/(4*epsi)*tgt2x**2)*0.002 + 0*tgt2y*VVt + sig*epsi**(-1/2)*VVt
    tgt2 = th.cat([tgt2xx,tgt2yy],dim = -1).reshape(out2.shape)
    aepre_loss = criterion(out1,tgt1)+criterion(out2,tgt2)
    return aepre_loss

# read data 
with h5py.File('./data/2BB_xy_1200_samples_enpredeesti%d.h5'% (10*sigma1), "r") as hf:
    dataset = hf['dataset'][:]
hf.close()

dataset = torch.Tensor(dataset)

dataset_ae = dataset[:,1:11,:]
import torch.utils.data as Data

train_dataset = Data.TensorDataset(dataset_ae)  # 先转换成 torch 能识别的 Dataset
data_loader = Data.DataLoader(
    dataset=train_dataset,      # torch TensorDataset format
    batch_size=batch_size,           # mini batch size 
    shuffle=True,          # shuffle on the batch
)

#training aepre
modelaepre = AE()
optimizer = th.optim.Adam(modelaepre.parameters(), lr=learning_rate)
#0-10~2-12
for epoch in range(num_epochs):   
    print('epoch {}'.format(epoch + 1))
    print('*' * 10)
    reconst_loss = 0.0
    for step, batch_xx in enumerate(data_loader):  # 每一步 loader 释放一小批数据用来学习
        #清除上一梯度
        optimizer.zero_grad()
        
        in_data = torch.Tensor([item.detach().numpy() for item in batch_xx]).reshape(-1,10,1200)
        aepredict = modelaepre(in_data)
        
        #loss = criterion(autoencoder, in_data)
        loss = aepre_loss(aepredict, in_data)
        
        #反向传播计算梯度
        loss.backward() 
        nn.utils.clip_grad_norm_(modelaepre.parameters(), 100.)
        optimizer.step() #应用梯度
        reconst_loss = loss.item()
        
        #print( epoch, step, batch_xx)
        if (step+1) % 2 == 0:
            print ("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}" 
                   .format(epoch+1, num_epochs, step+1, len(data_loader), reconst_loss))
aepredictl = modelaepre(dataset_ae.to(torch.float32))
        
#2-12~10-20
for ii in range(4):
    train_dataset1 = Data.TensorDataset(aepredictl)  # 先转换成 torch 能识别的 Dataset
    data_loader1 = Data.DataLoader(
        dataset=train_dataset1,      # torch TensorDataset format
        batch_size=batch_size,           # mini batch size 
        shuffle=True,          # shuffle on the batch
    )
    #training aepre
    for epoch in range(num_epochs):   # 训练所有!整套!数据次数
        print('epoch {}'.format(epoch + 1))
        print('*' * 10)
        reconst_loss = 0.0
        for step, batch_xx in enumerate(data_loader1):  # 每一步 loader 释放一小批数据用来学习
            #清除上一梯度
            optimizer.zero_grad()
            
            in_data = torch.Tensor([item.detach().numpy() for item in batch_xx]).reshape(-1,10,1200)
            aepredict = modelaepre(in_data)
            
            #loss = criterion(autoencoder, in_data)
            loss = aepre_loss(aepredict, in_data)
            
            #反向传播计算梯度
            loss.backward() 
            nn.utils.clip_grad_norm_(modelaepre.parameters(), 100.)
            optimizer.step() #应用梯度
            reconst_loss = loss.item()
            
            #print( epoch, step, batch_xx)
            if (step+1) % 2 == 0:
                print ("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}" 
                       .format(epoch+1, num_epochs, step+1, len(data_loader), reconst_loss))
            
    aepredictl = modelaepre(aepredictl.to(torch.float32))

#save pre_20
#save data
pre_20_save = aepredictl[:,:,:].detach().numpy()

fff = h5py.File('./data/2B_pre_20_%d.h5'% (10*sigma1),'w')
fff.create_dataset('pre_20', data=pre_20_save)
fff.close()
    
#10-20~30-40
for ii in range(5):
    train_dataset1 = Data.TensorDataset(aepredictl)  # 先转换成 torch 能识别的 Dataset
    data_loader1 = Data.DataLoader(
        dataset=train_dataset1,      # torch TensorDataset format
        batch_size=batch_size,           # mini batch size 
        shuffle=True,          # shuffle on the batch
    )
    #training aepre
    for epoch in range(num_epochs):   # 训练所有!整套!数据次数
        print('epoch {}'.format(epoch + 1))
        print('*' * 10)
        reconst_loss = 0.0
        for step, batch_xx in enumerate(data_loader1):  # 每一步 loader 释放一小批数据用来学习
            #清除上一梯度
            optimizer.zero_grad()
            
            in_data = torch.Tensor([item.detach().numpy() for item in batch_xx]).reshape(-1,10,1200)
            aepredict = modelaepre(in_data)
            
            #loss = criterion(autoencoder, in_data)
            loss = aepre_loss(aepredict, in_data)
            
            #反向传播计算梯度
            loss.backward() 
            nn.utils.clip_grad_norm_(modelaepre.parameters(), 100.)
            optimizer.step() #应用梯度
            reconst_loss = loss.item()
            
            #print( epoch, step, batch_xx)
            if (step+1) % 2 == 0:
                print ("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}" 
                       .format(epoch+1, num_epochs, step+1, len(data_loader), reconst_loss))
            
    aepredictl = modelaepre(aepredictl.to(torch.float32))
    
#save pre_30
#save data
pre_30_save = aepredictl[:,:,:].detach().numpy()

fff = h5py.File('./data/2B_pre_30_%d.h5'% (10*sigma1),'w')
fff.create_dataset('pre_30', data=pre_30_save)
fff.close()

for ii in range(5):
    train_dataset1 = Data.TensorDataset(aepredictl)  # 先转换成 torch 能识别的 Dataset
    data_loader1 = Data.DataLoader(
        dataset=train_dataset1,      # torch TensorDataset format
        batch_size=batch_size,           # mini batch size 
        shuffle=True,          # shuffle on the batch
    )
    #training aepre
    for epoch in range(num_epochs):   # 训练所有!整套!数据次数
        print('epoch {}'.format(epoch + 1))
        print('*' * 10)
        reconst_loss = 0.0
        for step, batch_xx in enumerate(data_loader1):  # 每一步 loader 释放一小批数据用来学习
            #清除上一梯度
            optimizer.zero_grad()
            
            in_data = torch.Tensor([item.detach().numpy() for item in batch_xx]).reshape(-1,10,1200)
            aepredict = modelaepre(in_data)
            
            #loss = criterion(autoencoder, in_data)
            loss = aepre_loss(aepredict, in_data)
            
            #反向传播计算梯度
            loss.backward() 
            nn.utils.clip_grad_norm_(modelaepre.parameters(), 100.)
            optimizer.step() #应用梯度
            reconst_loss = loss.item()
            
            #print( epoch, step, batch_xx)
            if (step+1) % 2 == 0:
                print ("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}" 
                       .format(epoch+1, num_epochs, step+1, len(data_loader), reconst_loss))
            
    aepredictl = modelaepre(aepredictl.to(torch.float32))
    
#save pre_40
pre_40_save = aepredictl[:,:,:].detach().numpy()
tru_40_save = dataset[:,40,:]
# Create a new file

fff = h5py.File('./data/2B_pre_40_%d.h5'% (10*sigma1),'w')
fff.create_dataset('pre_40', data=pre_40_save)
fff.create_dataset('tru_40', data=tru_40_save)
fff.close()
