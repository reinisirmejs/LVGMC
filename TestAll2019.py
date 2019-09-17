#Importing libraries for both making the training set and evaluating predictions from the model

import pandas as pd
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from keras.models import load_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np


def create_list(data, station = 'RIJE99PA'):
    df = data.loc[(data['EG_GH_ID'] == station)]

    df = df.to_numpy()

    df = df[:, 6:38]
    a = np.shape(df)[0]

    val = []
    for i in range(int(a/24)):
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

    #y3 = y3[~np.isnan(y3).any(axis=1)]
    return np.array(y3[:,0:look_back]), np.array(y3[:,look_back])

def create_training_example(TempVal,WindMag,WindDir,TempAns, look_back = 1):
    x1 = np.zeros((TempVal.shape[0], look_back, 3*len(location_set)))
    x1[:,:,0:len(location_set)] = (TempVal)
    x1[:,:,len(location_set):2*len(location_set)] = (WindMag*np.sin((WindDir/180)*np.pi))
    x1[:,:,2*len(location_set):3*len(location_set)] = (WindMag*np.cos((WindDir/180)*np.pi))
    index = []

    for i in range((x1.shape[0])):
        if np.isnan(x1[i]).any() == True or np.isnan(TempAns[i]).any() == True:
            index.append(i)
    index_full = np.linspace(0,x1.shape[0],x1.shape[0]+1)
    index_full = np.delete(index_full, index)
    print(index_full)

    x2 = np.zeros((len(index_full), look_back, 3*len(location_set)))
    x2ans = np.zeros((len(index_full),len(location_set)))
    for j in range(len(index_full)-1):
        x2[j,:,:] = x1[int(index_full[j]),:,:]
        x2ans[j,:] = TempAns[int(index_full[j]),:]
    return x2, x2ans

#reading the temperature data set

location_set = ['RIAI99PA','RIAL99MS','RIBA99PA','RIDM99MS','RIDO99MS','RIGU99MS','RIJE99PA','RIME99MS','RIPR99PA','RIRE99MS','RIRU99PA','RISA99PA','RISE99MS','RISI99PA','RIST99PA','RIVE99PA','RIZI99PA','RIZO99MS']

def create_all_dataset(x, look_back = 1, forecast = 1):
    list1 = create_list(x)
    length = len(list1)
    values = np.zeros((length-forecast-look_back,look_back,len(location_set)))
    answers = np.zeros((length-forecast-look_back, len(location_set)))

    for i in range(len(location_set)):
        values[:,:,i], answers[:,i] = create_dataset(create_list(x,station = location_set[i]), look_back = look_back, forecast = forecast)
    return values, answers

data = pd.read_csv("Avg_air_temperature_2019_6-8.csv")
data1 = pd.read_csv("Wind_direction_2019_6-8.csv")
data2 = pd.read_csv("Wind_speed_2019_6-8.csv")
#data2 = pd.read_csv("Wind_speed_2004.04_2018.csv")

#Scaling. NB Non-inverted data give much better results

look_back = 60
forecast =6
features  = 3*len(location_set)

valT, ansT = create_all_dataset(data, look_back = look_back, forecast = forecast)
valWM, ansWM = create_all_dataset(data2, look_back = look_back, forecast = forecast)
valWD, ansWD = create_all_dataset(data1, look_back = look_back, forecast = forecast)
Input_val, Ver_val = create_training_example(valT, valWM, valWD, ansT, look_back = look_back)

print("Input shape:",Input_val.shape)

X_train, X_test, Y_train, Y_test = train_test_split(Input_val, Ver_val, train_size=0.75, shuffle=True)

X_test1 = Input_val[0:400,:,:] #For plotting

Y_test1 = Ver_val[0:400]

###Calculate for
station = 'RIJE99PA'
id = location_set.index(station)

model = load_model('GRU200200NetworkAll6hW.h5')

train_predict = (model.predict(X_train))
test_predict = (model.predict(X_test))
test1_predict = (model.predict(X_test1))
all_predict = (model.predict(Input_val))

MAELoc = np.zeros(len(location_set))
RMSELoc = np.zeros(len(location_set))
for x in location_set:
    id = location_set.index(x)
    MAELoc[id] = mean_absolute_error(Ver_val[:,id], all_predict[:,id])
    RMSELoc[id] = np.sqrt(mean_squared_error(Ver_val[:,id], all_predict[:,id]))
    print(x)
    print('Train Mean Absolute Error:', mean_absolute_error(Y_train[:,id], train_predict[:,id]))
    print('Train Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_train[:,id], train_predict[:,id])))
    print('Test Mean Absolute Error:', mean_absolute_error(Y_test[:,id], test_predict[:,id]))
    print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_test[:,id], test_predict[:,id])))



#Plotting results
print('MAE', MAELoc)
print('RMSE', RMSELoc)
MAEAvg = np.sum(MAELoc)/18
RMSEAvg = np.sqrt(np.sum(RMSELoc**2)/18)

print('MAE Average', MAEAvg)
print('RMSE Average', RMSEAvg)

aa=[x for x in range(400)]
plt.plot(aa, Y_test1[:,id], marker='.', label="actual")
plt.plot(aa, test1_predict[:,id], 'r', label="prediction")
plt.tight_layout()
plt.subplots_adjust(left=0.07)
plt.ylabel('Temperature', size=15)
plt.xlabel('Time step', size=15)
plt.legend(fontsize=15)
plt.show()
