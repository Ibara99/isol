import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#namafile = r"E:\fais\UtmNew\Skripsi\data\datareal1jam.csv"
namafile = r"E:\fais\UtmNew\Skripsi\data\datareal1jam2.csv"
data = pd.read_csv(namafile)

dataAsli = data.to_numpy()[:, 1:] #timestamp diilangin
scaler = MinMaxScaler()
scaler.fit(dataAsli)
data = scaler.transform(dataAsli)

dt = []
tmp = data
w = 8
for t, row in enumerate(tmp):
  if t > w:
    r = []
    for i in range(w):
      r.insert(0, tmp[t-i][0])
    dt.append(r)
data = np.array(dt)

#X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.33, random_state=1)
X_train, X_test, y_train, y_test = data[:300,:w-1], data[300:,:w-1], data[:300,w-1], data[300:,w-1]

regr = MLPRegressor(random_state=1, max_iter=1000).fit(X_train, y_train)

pred = regr.predict(X_test)

denormp = (pred * dataAsli.max() - pred * dataAsli.min()) + dataAsli.min()
denormy = (y_test * dataAsli.max() - y_test * dataAsli.min()) + dataAsli.min()

MSE = mean_squared_error(denormp, denormy)

plt.plot([i for i in range(len(denormp))], denormp, color='b')
plt.plot([i for i in range(len(denormy))], denormy, color='g')
plt.show()
