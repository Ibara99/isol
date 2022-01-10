# Note: Please install following libraries to run the flask API : flask, pymongo, dnspython, flask_cors
# > set FLASK_APP=flaskr
# > set FLASK_ENV=development
# > flask run

from flask import Flask, jsonify, request, render_template, redirect
from flask_cors import CORS
##import pymongo
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pickle
  
# connection_url = 'mongodb+srv://admin:samplearticle@cluster0-pm5vp.mongodb.net/test?retryWrites=true&w=majority'
# connection_url = 'mongodb+srv://ibara1010:admin123@cluster0.4gh6a.mongodb.net/test?retryWrites=true&w=majority'
##connection_url = 'mongodb://ibara1010:admin123@cluster0-shard-00-00.4gh6a.mongodb.net:27017,cluster0-shard-00-01.4gh6a.mongodb.net:27017,cluster0-shard-00-02.4gh6a.mongodb.net:27017/test?ssl=true&replicaSet=atlas-i9oz7l-shard-0&authSource=admin&retryWrites=true&w=majority'

app = Flask(__name__)
# client = pymongo.MongoClient(connection_url)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Kalau butuh define sendiri :  
# Database = client["test"]
# SampleTable = Database["iot__"]

# # Database
# Database = client.get_database('Example')
# # Table
# SampleTable = Database.SampleTable
file = r"E:\fais\UtmNew\Skripsi\data\datareal1jam2.csv"
data = pd.read_csv(file)
tgl, data = data.to_numpy()[:,0], data.to_numpy()[:,1:]
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
tgltrain, tgltest = tgl[:], tgl[300:]
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

# Ini home
@app.route('/', methods=['GET'])
def home():
##    tmp = moving_average(dataPh, 100)
##    return "PH</br> MAD "+str(MAD(dataPh[99:],tmp))+"</br> </br> MSE "+str(MSE(dataPh[99:],tmp))+"</br> </br> MAPE "+str(MAPE(dataPh[99:],tmp))+"</br> </br> </br> Sal </br>"+"MAD "+str(MAD(dataSal[99:],tmp))+"</br> </br> MSE "+str(MSE(dataSal[99:],tmp))+"</br> </br> MAPE "+str(MAPE(dataSal[99:],tmp))
    return ""

#coba render
@app.route('/dashboard/', methods=['GET'])
def dashboard():
    return render_template('Dashboard.html')

@app.route('/dataview/', methods=['GET'])
def dataview():
    return render_template('dataview.html')

@app.route('/about/', methods=['GET'])
def about():
    return render_template('about.html')

@app.route('/backpro/', methods=['GET'])
def backpro():
    return render_template('backpro.html')
@app.route('/apibackpro/', methods=['GET'])
def apibackpro():
    #train(xtrain, ytrain)
    denorm, MSE = test(xtest,ytest)
    tmp = []
    for row in range(len(denorm)):
        tmp.append({"timestamp":tgltest[row], "ph":denorm[row], "asli":ytest[row]})
    return jsonify(tmp)
  
# To find all the entries/documents in a table/collection,
# find() function is used. If you want to find all the documents
# that matches a certain query, you can pass a queryObject as  
# argument.
'''
@app.route('/find/', methods=['GET'])
def findAll():
    query = SampleTable.find()
    output = {}
    i = 0
    for x in query:
        output[i] = x
        output[i].pop('_id')
        i += 1
    return jsonify(output)
'''
@app.errorhandler(404)
def not_found(error):
    url = request.path
    if url.islower():
        return "not found"
    else:
        return redirect(request.full_path.lower())

if __name__ == '__main__':
    app.run(debug=True)
