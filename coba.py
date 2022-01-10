import numpy as np
import pandas as pd
from random import random
import matplotlib.pyplot as plt
import pickle

file = r"E:\fais\UtmNew\Skripsi\data\datareal1jam2.csv"
data = pd.read_csv(file)
data = data.to_numpy()[:,1:]
norm = (data-data.min(axis=0))/(data.max(axis=0)-data.min(axis=0))

dt = []
tmp = norm
w = 8
for t, row in enumerate(tmp):
  if t > w:
    r = []
    for i in range(w):
      r.insert(0, tmp[t-i][0])
    dt.append(r)
norm = np.array(dt)

xtrain, xtest, ytrain, ytest = norm[:,:w-1], norm[300:,:w-1], norm[:,w-1], norm[300:,w-1]
nInput = norm.shape[1]-1
nHidden = 100
nOutput = 1
iterasi = 1000
alpha = 0.0001#learningRate
TolErr = 0.00001
v = np.random.rand(nInput+1, nHidden)
w = np.random.rand(nHidden+1, nOutput)

def train(xtrain, ytrain):
    global v, w, data
    counter = 0
    MSE = TolErr + 1
    y = (ytrain*data.max()-ytrain*data.min())+data.min()
    while counter < iterasi and MSE > TolErr:
        pred = []
        for t in range(len(xtrain)):#inputhidden
            x = xtrain[t]
            z = []
            for j in range(nHidden):#tiap titik di hidden layer
                inputhidden = 0
                for i in range(nInput):
                    inputhidden += x[i]*v[i,j]
                inputhidden += 1*v[-1,j]
                #inputhidden = 1/(1+np.exp(inputhidden))#fungsigmoid
                inputhidden = max(0, inputhidden)
                z.append(inputhidden)
            o = 0#hiddenoutput
            for j in range(nHidden):
                o += z[j]*w[j,0]
            o += 1*w[-1,0]
            #o = 1/(1+np.exp(o))#fungsigmoid
            pred.append(o)
            #backpro(delta)
            a = []#deltaK
            a = (ytrain[t]-o)*1
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
                if z[j] >= 0:
                  turunan = 1
                else :
                  turunan = 0
                deltaj.append(deltaj_inj*turunan)#z[j]*(1-z[j]))
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
        pred = np.array(pred)
        denorm = (pred*data.max()-pred*data.min())+data.min()
        MSE = 0
        for t in range(len(xtrain)):
            MSE += (y[t]-denorm[t])**2
        MSE = MSE / len(xtrain)
        counter += 1
##    vw = {}
##    vw["v"] = v
##    vw["w"] = w
##    vwfile = open("vwfile", "wb")
##    pickle.dump(vw, vwfile)
##    vwfile.close()
    fig, ax = plt.subplots()
    ax.plot([i for i in range(len(denorm))], denorm, color='b')
    ax.plot([i for i in range(len(y))], y, color='g')
    fig.show()
    return (v,w,denorm,MSE)
def loadData():
    vwfile = open("vwfile", "rb")
    vw = pickle.load(vwfile)
    tmp ={}
    for keys in vw:
        #print(keys, vw[keys])
        tmp[keys]= vw[keys]
    vwfile.close()
    return tmp['v'], tmp['w']
def test(xtest, ytest):
    v,w = loadData()
    pred = []
    ytest = (ytest*data.max()-ytest*data.min())+data.min()
    for t in range(len(xtest)):
        x = xtest[t] #inputHidden
        z = []
        for j in range(nHidden):
            inputhidden = 0
            for i in range(nInput):
                inputhidden += x[i]*v[i,j]
            inputhidden += 1*v[-1,j]
            inputhidden = max(0, inputhidden)
            #inputhidden = 1/(1+np.exp(inputhidden))
            z.append(inputhidden)
        o = 0 #hiddenoutput
        for j in range(nHidden):
            o += z[j]*w[j,0]
        o += 1*w[-1,0]
        #o = 1/(1+np.exp(o))
        #o = max(0,o)
        pred.append(o)
    pred = np.array(pred)
    denorm = (pred*data.max()-pred*data.min())+data.min()
    MSE = 0
    for t in range(len(xtest)):
        MSE += (ytest[t]-denorm[t])**2
    MSE = MSE/len(xtest)
    fig, ax = plt.subplots()
    ax.plot([i for i in range(len(denorm))], denorm, color='b')
    ax.plot([i for i in range(len(ytest))], ytest, color='g')
    fig.show()
    return (denorm, MSE)

##a = train(xtrain,ytrain)
b = test(xtest,ytest)

##print(b)
