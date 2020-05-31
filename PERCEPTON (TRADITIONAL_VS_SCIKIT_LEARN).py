#!/usr/bin/env python
# coding: utf-8

# In[17]:


import time
import pandas as pd
import numpy as np
train_set = pd.read_csv('C:\\Users\\KIRAN KONDISETTI\\Desktop\\WineData.csv')
test_set = pd.read_csv('C:\\Users\\KIRAN KONDISETTI\\Desktop\\WineHoldoutData.csv')


# In[2]:


train = train_set.drop('quality',axis=1)
train
for i in range(len(train_set.iloc[:,0])):
    if(train_set.quality[i]>6):
        train_set.quality[i] = 1
    else:
        train_set.quality[i] = 0
target = train_set.iloc[:,11]  
train1 = test_set.drop('quality',axis=1)
for i in range(len(test_set.iloc[:,0])):
    if(test_set.quality[i]>6):
        test_set.quality[i] = 1
    else:
        test_set.quality[i] = 0
target1 = test_set.iloc[:,11] 
train2 = train.drop('style',axis=1)
train_set.head(20)
X = np.array(train2)
X1 = np.array(train1.drop('style', axis = 1))
Y = np.array(target)
Y1 = np.array(target1)


# In[18]:


def predict_wthreshold(row, weights, threshold):
    activation = weights[0]
    for i in range(len(row)-1):
        activation += weights[i + 1] * row[i]
        return 1.0 if activation >= threshold else 0.0

def train_weights_wthreshold(train, target, l_rate, n_epoch, threshold):
    weights = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        sum_error = 0.0
        for i, row in enumerate(train):
            prediction = predict_wthreshold(row, weights, threshold)
            error = int(target[i]) - prediction
            sum_error += error**2
            weights[0] = weights[0] + l_rate * error
            for i in range(len(row)-1):
                weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
                return weights
start = 0
stop = 0
start = time.time() 
weights = train_weights_wthreshold(X, Y, 1,11, 0)
errors = 0
for i in range(len(X1)):
    prediction = predict_wthreshold(X1[i], weights, 0)
    if (prediction != int(Y1[i])):
        errors += 1
stop = time.time()
print(f"Time Taken by traditional algorithm: {stop - start}s")
accuracy_tr= 1 - (errors / len(X1)) # Checking for errors and computing score
accuracy_tr


# In[22]:


from sklearn import metrics
from sklearn.linear_model import Perceptron
clf = Perceptron(tol=1e-3, random_state=1)
start = 0
stop = 0
start = time.time() 
clf.fit(X, Y)
predict = clf.predict(X1)
predict

errors=0
for i in range(len(X1)):
    
    if (predict[i] == int(Y1[i])):
        errors += 1
stop = time.time()
print(f"Time Taken by scikit algorithm: {stop - start}s")
accuracy_sk = 1-(errors / len(X1)) # Checking for errors and computing score
accuracy_sk


# In[ ]:





# In[ ]:




