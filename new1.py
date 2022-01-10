import numpy as np
import pandas as pd
from random import random

namafile = r"E:\fais\UtmNew\Skripsi\data\dataperjam.xlsx"
data = pd.read_excel(namafile, sheet_name="Sheet1")
print(data)
#data = data.to_numpy()
data = data.to_numpy()[:,:-1]
dtNormalisasi = (data-data.min(axis=0))/(data.max(axis=0)-data.min(axis=0))


nInput = data.shape[1]-1
nHidden = 10
nOutput = 1
alpha =  0.00001 #0.2#learning rate
toleransiEror = 0.00001
iterasi = 500
v = np.random.rand(nInput+1, nHidden)
w = np.random.rand(nHidden+1,nOutput)
y = dtNormalisasi[:, -1]#ydata

counter = 0
MSE = toleransiEror + 1
while counter < iterasi and MSE > toleransiEror:
    Frediksi=[]
    for t in range(len(dtNormalisasi)):
        #inputHidden
        x = dtNormalisasi[t]
        z = []
        for j in range(nHidden): # tiap titik di hidden layer
            inputHidden = 0
            for i in range(nInput):
                inputHidden += x[i]*v[i,j]
            inputHidden += 1*v[-1, j]
            inputHidden = 1/(1+np.exp(inputHidden))#fungsigmoid
            z.append(inputHidden)
        #hiddenOutput
        o=0
        for j in range(nHidden):
            o += z[j]*w[j,0]
        o += 1*w[-1,0]
        o = 1/(1+np.exp(o))#fungsigmoid
        Frediksi.append(o)
        #backpro(delta)
        a = []#deltaK
        a = (y[t]-o)*o*(1-o)
        #backpro(koreksibobot/delta)
        deltaW = []#Wjk
        for k in range(nHidden):
            deltaW.append([alpha*a*z[j]])
        deltaW.append([alpha*a])
        deltaj = []#deltaHiddenInput
        deltaVij = []
        for j in range(nHidden):
            deltaj_inj = 0
            for k in range(nInput):
                deltaj_inj += a*v[k][j]
            deltaj.append(deltaj_inj*z[j]*(1-z[j]))   
        for i in range(nInput):
            tmp = []
            for j in range(nHidden):
                tmp.append(alpha*deltaj[j]*x[i])
            deltaVij.append(tmp)
        tmp = []
        for j in range(nHidden):
            tmp.append(alpha*deltaj[j])
        deltaVij.append(tmp)
        deltaVij = np.array(deltaVij)
        deltaW = np.array(deltaW)
        w = w + deltaW
        v = v + deltaVij
    
    Frediksi = np.array(Frediksi)
    denormalisasi = (Frediksi * data.max() - Frediksi * data.min()) + data.min()    
    #ngitungeror
    MSE = 0
    for t in range(len(dtNormalisasi)):
        MSE += (y[t]-Frediksi[t])**2
    MSE = MSE / len(dtNormalisasi)
    counter += 1
    print(counter, MSE)
print("Hasil Prediksi")
print(Frediksi)
print("Denormalisasi")
print(denormalisasi)
#Learning
print("Hasil Learning")
namafile = r"E:\fais\UtmNew\Skripsi\data\dataperjam.xlsx"
Data = pd.read_excel(namafile, sheet_name="Sheet1")
Data = Data.to_numpy()[:,:-1]
dtNormalisasi = (Data-Data.min(axis=0))/(Data.max(axis=0)-Data.min(axis=0))
y = dtNormalisasi[:, -1]#ydata

Prediksi=[]
for t in range(len(dtNormalisasi)):
    #inputHidden
    x = dtNormalisasi[t]
    
    z = []
    for j in range(nHidden): # tiap titik di hidden layer
        inputHidden = 0
        for i in range(nInput):
            inputHidden += x[i]*v[i,j]
        inputHidden += 1*v[-1, j]

        inputHidden = 1/(1+np.exp(inputHidden))#fungsigmoid
        z.append(inputHidden)
        
    #hiddenOutput
    o=0
    for j in range(nHidden):
        o += z[j]*w[j,0]
    o += 1*w[-1,0]
    o = 1/(1+np.exp(o))#fungsigmoid
    Prediksi.append(o)
Prediksi = np.array(Prediksi)
denormalisasi = (Prediksi * Data.max() - Prediksi * Data.min()) + Data.min()
MSE = 0
for t in range(len(dtNormalisasi)):
    MSE += (Data[t,-1]-denormalisasi[t])**2
MSE = MSE / len(dtNormalisasi)
print(MSE)
#Prediksi = np.array(Prediksi)
print("Hasil Prediksi")
print(Prediksi)
#denormalisasi = (Prediksi * Data.max() - Prediksi * Data.min()) + Data.min()
print("Denormalisasi")
print(denormalisasi)
