import pandas as pd
pd.set_option('display.float_format', lambda x: '%.4f' % x)
import numpy as np
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def Tscaler(x):
    return (x+30)/75

def TInvScaler(x):
    return x*75-30

def create_dataset(dataset, look_back=100, forecast = 24):
    X, Y = [], []
    for i in range(len(dataset)-look_back-forecast):            #creates the data set exapmles, with look_back previous values forecasting forecast hours ahead
        a = dataset[i:(i+look_back)]
        X.append(a)
        Y.append(dataset[i + look_back+forecast-1])

    y3 = np.zeros((len(dataset)-look_back-forecast, look_back + 1))             #gets rid of all the training examples that contain nan
    for j in range(len(dataset)-look_back-forecast):
        y3[j, 0:look_back] = X[j]
        y3[j,look_back] = Y[j]

    y4 = y3[~np.isnan(y3).any(axis=1)]
    return np.array(y4[:,0:look_back]), np.array(y4[:,look_back])

#reading the temperature data set

data = pd.read_csv("Avg_air_temperature_2004.04_2018.csv")
station = 'RIJE99PA'

#Creates a list of temperature values

df = data.loc[(data['EG_GH_ID']==station)]

df = df.to_numpy()

df= df[:, 6:38]
val = []
for i in range(175):
    for j in range(31):
        y1 = df[i*24:i*24+24,j]
        val.append(y1)

val = np.concatenate((val), axis = None)

#Scaling. NB Non-inverted data give much better results

#val = Tscaler(val)

look_back = 60
forecast = 1

Input_val, Ver_val = create_dataset(val, look_back = look_back, forecast = forecast)

Input_valdf = pd.DataFrame(Ver_val)

export_csv = Input_valdf.to_csv(r"C:\Users\Reinis\Desktop\TVerSet.csv", index = None, header=False)

X_train, X_test, Y_train, Y_test = train_test_split(Input_val, Ver_val, train_size=0.75, shuffle=True)
X_test1  = Input_val[10000:10400]
X_test1 = np.reshape(X_test1, (X_test1.shape[0], X_test1.shape[1], 1))
Y_test1 = Ver_val[10000:10400]

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

print("Input data shape:", X_train.shape, Y_train.shape)

print(X_train[0:5])

#LSTM model

model = Sequential()
model.add(GRU(50, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences= True))
model.add(Dropout(0.2))
model.add(GRU(50))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

history = model.fit(X_train, Y_train, epochs=50, batch_size=500, validation_data= (X_test, Y_test),
                    callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=1, shuffle=True)

model.summary()

model.save('GRU5050NetworkSingle1h.h5')

train_loss = history.history['loss']
val_loss = history.history['val_loss']

count_epochs = range(1,len(val_loss)+1)

train_predict = (model.predict(X_train))
test_predict = (model.predict(X_test))
test1_predict = (model.predict(X_test1))


# invert predictions

#Y_train = TInvScaler(Y_train)
#Y_test = TInvScaler(Y_test)
#Y_test1 = TInvScaler(Y_test1)

#train_predict = TInvScaler(model.predict(X_train))
#test_predict = TInvScaler(model.predict(X_test))
#test1_predict = TInvScaler(model.predict(X_test1))


print('Train Mean Absolute Error:', mean_absolute_error(Y_train, train_predict))
print('Train Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_train, train_predict)))
print('Test Mean Absolute Error:', mean_absolute_error(Y_test, test_predict))
print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_test, test_predict)))

#Plotting results

plt.plot(count_epochs,train_loss)
plt.plot(count_epochs,val_loss)
plt.legend(['Train loss', 'Validation loss'])
plt.show()

aa=[x for x in range(400)]
plt.plot(aa, Y_test1, marker='.', label="actual")
plt.plot(aa, test1_predict, 'r', label="prediction")
plt.tight_layout()
plt.subplots_adjust(left=0.07)
plt.ylabel('Temperature', size=15)
plt.xlabel('Time step', size=15)
plt.legend(fontsize=15)
plt.show()

