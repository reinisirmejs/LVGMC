#Here we import all the necessary libraries, pandas, sklearn and numpy for data pre-processing, keras for neural networks
# and matplotlib in order to plot the obtained data

import pandas as pd
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

#This first function is specific to the datasets that I used, it takes the data of a given station and creates a list of
# values. NB! Make sure that the file contains all the entries even if they're NaN

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

#This function takes a list as created using create_list and creates 2 arrays
# one of the arrays contain prediction values going back forecast steps
# the other is a list containing answers that are forecast steps ahead from the last training entry
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
    return np.array(y3[:,0:look_back]), np.array(y3[:,look_back])

#The first part of this function takes 3 datasets and puts them together
#The second part finds all the entries that are NaN either for prediction data or answers and removes the rows containing
#them
#The output is a ready to process dataset with answers for one of the stations

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

#Just a list of weather stations that are gonna be used

location_set = ['RIAI99PA','RIAL99MS','RIBA99PA','RIDM99MS','RIDO99MS','RIGU99MS','RIJE99PA','RIME99MS','RIPR99PA','RIRE99MS','RIRU99PA','RISA99PA','RISE99MS','RISI99PA','RIST99PA','RIVE99PA','RIZI99PA','RIZO99MS']

#This function creates a dataset for all the weather stations, it combines create_list and create_dataset functions
#and creates a ready to use dataset with answers to input into a recurrent network. Recurrent networks have a particular
#shape that they need to take in order to be correctly read [samples, timesteps, features]. This function also creates
#it into one. As an output it gives [3D trianing array, 2D answer array]

def create_all_dataset(x, look_back = 1, forecast = 1):
    list1 = create_list(x)
    length = len(list1)
    values = np.zeros((length-forecast-look_back,look_back,len(location_set)))
    answers = np.zeros((length-forecast-look_back, len(location_set)))

    for i in range(len(location_set)):
        values[:,:,i], answers[:,i] = create_dataset(create_list(x,station = location_set[i]), look_back = look_back, forecast = forecast)
    return values, answers

#pandas read_csv function is used to read the csv files

data = pd.read_csv("Avg_air_temperature_2004.04_2018.csv")
data1 = pd.read_csv("Wind_direction_2004.04_2018.csv")
data2 = pd.read_csv("Wind_speed_2004.04_2018.csv")

#NB! In here it's necessary to enter the value how many hours ahead the forecasting is performed

look_back = 60
forecast = 12
features = 3*len(location_set)

#Here one can choose the station for which the plot is going to be made

station = 'RIJE99PA'
id = location_set.index(station)

#Here we create datasets and answers for Temperature, Wind speed and direction

valT, ansT = create_all_dataset(data, look_back = look_back, forecast = forecast)
valWM, ansWM = create_all_dataset(data2, look_back = look_back, forecast = forecast)
valWD, ansWD = create_all_dataset(data1, look_back = look_back, forecast = forecast)

#Here all 3 datasets are merged together into a ready to use dataset

Input_val, Ver_val = create_training_example(valT, valWM, valWD, ansT, look_back = look_back)

print("Input shape:",Input_val.shape)

#Train_test_split function splits the training set into a training set and verification set and shuffles the examples
#To improve the testing accuracy

X_train, X_test, Y_train, Y_test = train_test_split(Input_val, Ver_val, train_size=0.75, shuffle=True)

#Here we hand-pick some unshuffled values for plotting

X_test1 = Input_val[10000:10400,:,:]
Y_test1 = Ver_val[10000:10400]

print("Training set made")

#After this we have finished with the data pre-processing and we define our neural network model

#GRU model that is used for predictions

model = Sequential()
model.add(GRU(200, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences= True))
model.add(Dropout(0.2))
model.add(GRU(200))
model.add(Dropout(0.2))
model.add(Dense(len(location_set)))
model.compile(loss='mean_squared_error', optimizer='adam')

#The following line fits the training data to the model and compiles the results with the given batch size and the number
#of epochs

history = model.fit(X_train, Y_train, epochs=50, batch_size=500, validation_data=(X_test, Y_test),
                        callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=1, shuffle=True)

model.summary()

#This line saves the model. NB! Make sure to name the model accordingly!!!

model.save('GRU200200NetworkAll12hW.h5')

#Obtaining the loss values

train_loss = history.history['loss']
val_loss = history.history['val_loss']

count_epochs = range(1,len(val_loss)+1)

#Making the predictions  on the ready-made model for both training and test data

train_predict = (model.predict(X_train))
test_predict = (model.predict(X_test))
test1_predict = (model.predict(X_test1))


print('Train Mean Absolute Error:', mean_absolute_error(Y_train[:,id], train_predict[:,id]))
print('Train Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_train[:,id], train_predict[:,id])))
print('Test Mean Absolute Error:', mean_absolute_error(Y_test[:,id], test_predict[:,id]))
print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_test[:,id], test_predict[:,id])))

#Plotting loss results

plt.plot(count_epochs,train_loss)
plt.plot(count_epochs,val_loss)
plt.legend(['Train loss', 'Validation loss'])
plt.show()

#Plotting validation results

aa=[x for x in range(400)]
plt.plot(aa, Y_test1[:,id], marker='.', label="actual")
plt.plot(aa, test1_predict[:,id], 'r', label="prediction")
plt.tight_layout()
plt.subplots_adjust(left=0.07)
plt.ylabel('Temperature', size=15)
plt.xlabel('Time step', size=15)
plt.legend(fontsize=15)
plt.show()
