import pandas as pd
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping
from keras.models import load_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

def Tscaler(x):
    return (x+30)/75

def TInvScaler(x):
    return x*75-30

def Wscaler(x):
    return (x+20)/40

def WInvscaler(x):
    return x*40-20

def create_list(data, station = 'RIJE99PA'):
    df = data.loc[(data['EG_GH_ID'] == station)]

    df = df.to_numpy()

    df = df[:, 6:38]

    val = []
    for i in range(175):
        for j in range(31):
            y1 = df[i*24:i*24 + 24, j]
            val.append(y1)
    return np.concatenate((val), axis = None)

#[Temp, windSpeed, windDir]

def create_dataset(dataset, look_back=1, forecast = 1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-forecast):            #creates the data set exapmles, with look_back previous values forecasting forecast hours ahead
        a = dataset[i:(i+look_back)]
        X.append(a)
        Y.append(dataset[i + look_back+forecast-1])

    y3 = np.zeros((len(dataset)-look_back-forecast, look_back + 1))             #gets rid of all the training examples that contain nan
    for j in range(len(dataset)-look_back-forecast):
        y3[j, 0:look_back] = X[j]
        y3[j,look_back] = Y[j]

    y3 = y3[~np.isnan(y3).any(axis=1)]
    return np.array(y3[:,0:look_back]), np.array(y3[:,look_back])

def create_training_example(TempVal,WindMag,WindDir,TempAns, look_back = 1):
    x1 = np.zeros((TempVal.shape[0], look_back, 3))
    x1[:,:,0] = (TempVal)
    x1[:,:,1] = (WindMag*np.sin((WindDir/180)*np.pi))
    x1[:,:,2] = (WindMag*np.cos((WindDir/180)*np.pi))
    index = []

    for i in range((x1.shape[0])):
        if np.isnan(x1[i]).any() == True or np.isnan(TempAns[i]).any() == True:
            index.append(i)
    index_full = np.linspace(0,x1.shape[0],x1.shape[0]+1)
    index_full = np.delete(index_full, index)
    print(index_full)

    x2 = np.zeros((len(index_full), look_back, 3))
    x2ans = np.zeros((len(index_full)))
    for j in range(len(index_full)-1):
        x2[j,:,:] = x1[int(index_full[j]),:,:]
        x2ans[j] = TempAns[int(index_full[j])]
    return x2, x2ans

#reading the temperature data set

location_set = ['RIAI99PA','RIAL99MS','RIBA99PA','RIDM99MS','RIDO99MS','RIGU99MS','RIJE99PA','RIME99MS','RIPR99PA','RIRE99MS','RIRU99PA','RISA99PA','RISE99MS','RISI99PA','RIST99PA','RIVE99PA','RIZI99PA','RIZO99MS']

def create_all_dataset(x, look_back = 1, forecast = 1):
    values = np.zeros((130200-forecast-look_back,look_back,len(location_set)))
    answers = np.zeros((130200-forecast-look_back, len(location_set)))
    for i in range(len(location_set)):
        values[:,:,i], answers[:,i] = create_dataset(create_list(x,station = location_set[i]), look_back = look_back, forecast = forecast)
    return values, answers


#reading the temperature data set

location_set = ['RIAI99PA','RIAL99MS','RIBA99PA','RIDM99MS','RIDO99MS','RIGU99MS','RIJE99PA','RIME99MS','RIPR99PA','RIRE99MS','RIRU99PA','RISA99PA','RISE99MS','RISI99PA','RIST99PA','RIVE99PA','RIZI99PA','RIZO99MS']


data = pd.read_csv("Avg_air_temperature_2004.04_2018.csv")
data1 = pd.read_csv("Wind_direction_2004.04_2018.csv")
data2 = pd.read_csv("Wind_speed_2004.04_2018.csv")
station = 'RIJE99PA'

#Creates a list of temperature values

valT = create_list(data, station = station)
print(valT.shape)
valWS = create_list(data2, station = station)
valWD = create_list(data1, station = station)

#valWX, valWY = valWS*np.sin((valWD/180)*np.pi), valWS*np.cos((valWD/180)*np.pi)

#Scaling. NB Non-inverted data give much better results

#val = Tscaler(val)

look_back = 60
forecast = 1
features  = 1

Input_valT, Ver_valT = create_dataset(valT, look_back = look_back, forecast = forecast)

X_train, X_test, Y_train, Y_test = train_test_split(Input_valT, Ver_valT, train_size=0.75, shuffle=True)

X_test1 = Input_valT[3000:3400] #For plotting
X_test1 = np.reshape(X_test1, (X_test1.shape[0], X_test1.shape[1], 1))
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

Y_test1 = Ver_valT[3000:3400]

###Calculate for
station = 'RIJE99PA'
id = location_set.index(station)

model = load_model('GRU5050NetworkSingle1h.h5')

X_test2 = X_test1[0]
X_test2 = np.reshape(X_test2, (1, X_test2.shape[0], X_test2.shape[1]))
print(X_test2.shape)
test_predict = np.zeros(100)
for i in range(100):
    test_predict[i] = (model.predict(X_test2))
    X_test2 = np.delete(X_test2, (0))
    X_test2 = np.append(X_test2, (test_predict[i]))
    X_test2 = np.reshape(X_test2, (1, look_back, 1))


test1_predict = (model.predict(X_test1))

#Plotting results

aa=[x for x in range(400)]
ab = [x for x in range(100)]
plt.plot(aa, Y_test1, marker='.', label="actual")
plt.plot(aa, test1_predict, 'r', label="prediction")
plt.plot(ab,test_predict, label = "forecast")
plt.tight_layout()
plt.subplots_adjust(left=0.07)
plt.ylabel('Temperature', size=15)
plt.xlabel('Time step', size=15)
plt.legend(fontsize=15)
plt.show()