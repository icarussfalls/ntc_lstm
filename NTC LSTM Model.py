#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
from pandas_datareader import data as pdr
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# In[2]:


#get the web quoted
df = pd.read_csv('/home/icarus/Downloads/NTC.csv', index_col = 'Date', parse_dates=True)
df


# In[3]:


#visualize the closing price hostory
plt.figure(figsize=(16,8))
plt.title('Close price history')
plt.plot(df['Price'])
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Close Price USD ($)', fontsize = 18)
plt.show()


# In[4]:


#create a new dataframe with only the close column
data = df.filter(['Price'])
#convert the dataframe to a numpy array
dataset = data.values
#get the number of rows to train the model on
training_data_len = round(len(dataset)*  .8)
training_data_len


# In[5]:


#scale the data
#apply preprocessing input datas before feeding to lstm model
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset) #the range will be 0 to 1 inclusive
scaled_data


# In[39]:


#create the training data set
#create the scaled training data set
train_data = scaled_data[0:training_data_len , :]
#split the data into x_train and y_train
x_train = []
y_train = []

for i in range(60,len(train_data)):
  x_train.append(train_data[i-60:i, 0])
  y_train.append(train_data[i, 0])
  if i<=61:
    print(x_train)
    print(y_train)
    print()


# In[7]:


#convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)


# In[8]:


#reshaoe the data lstm expecting three dimensional data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape


# In[26]:


# build the lstm model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape = (x_train.shape[1], 1 )))
model.add(LSTM(50, return_sequences = False))
model.add(Dense(25))
model.add(Dense(1))


# In[27]:


#complile the model
model.compile(optimizer='adam', loss='mean_squared_error')


# In[29]:


#train the model
model.fit(x_train, y_train, batch_size =1, epochs = 25)


# In[31]:


#create the testing data settest_data = scaled_data[training_data_len - 60:, :]
#create the data set x_test and y_test
x_test = []
y_test= dataset[training_data_len:, :]
for i in range (60, len(test_data)):
  x_test.append(test_data[i-60:i, 0])


# In[32]:


#convert the data to a numpy array
x_test = np.array(x_test)


# In[33]:


#reshape the data as data set needs to be three dimensional
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))


# In[34]:


#get the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)


# In[38]:


#get the root mean squared error(RMSE)
rmse = np.sqrt( np.mean(predictions - y_test)**2)
rmse


# In[36]:


#plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
#visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Close Price', fontsize = 18)
plt.plot(train['Price'])
plt.plot(valid[['Price', 'Predictions' ]])
plt.legend(['Train', 'Val', 'Predictions'], loc= 'lower right')
plt.show()


# In[37]:


#get the quote
quote = pd.read_csv('/home/icarus/Downloads/NTC.csv', index_col = 'Date', parse_dates=True)
#create a new dataframe
a = quote.filter(['Price'])
#get the last 60 days closing values and convert the dataframe to an array
last_60_days = a[-60:].values
#scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)
#create an empty list
X_test = []
#append the past 60 days to X_test list
X_test.append(last_60_days_scaled)
#convert the X_test data set to a numpy array
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)
#predicts next day price



