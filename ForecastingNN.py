import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

#define functions that were used to scale the temperature data

def Tscaler(x):
    return (x+30)/75

def TscalerInv(x):
    return x*75-30

df = pd.read_csv('TSet6-7.csv')      #read the data file

x = df[df.columns[:54]]
y = df[df.columns[54:72]]
x = Tscaler(x)
y = Tscaler(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.75,shuffle=False) #assign the train and test sets

#Create the NN

model = tf.keras.Sequential([
    tf.keras.layers.Dense(54, activation = 'linear', input_dim = 54),
    tf.keras.layers.Dense(250, activation = 'linear'),
    tf.keras.layers.Dense(250, activation = 'linear'),
    tf.keras.layers.Dense(18, activation = 'linear')
])

#Compiling and predictions

model.compile(optimizer = 'adam',
              loss = 'mean_squared_error')

hist1 = model.fit(
    x_train, y_train,
    epochs = 300, batch_size = 1000, #batch size for 1h = 55,
    validation_data=(x_test,y_test),
    shuffle=True)   #fitting test data to the model

model.save('forecast.h5')

ans = model.predict([x_test])
ans = TscalerInv(ans)            #lines for scaled model
ansReal = TscalerInv(y_test) #lines for scaled model
ansReal = ansReal.to_numpy()
x_test1 = x_test[x_test.columns[0:18]]
x_test1 = TscalerInv(x_test1)

train_loss = hist1.history['loss']
val_loss = hist1.history['val_loss']
epoch_count = range(1, len(train_loss) + 1)

#Calculation of errors for comparison

temp_err = np.sqrt(mean_squared_error(ans, ansReal)) #calculate the temperature error for the model
temp_err2 = np.sqrt(mean_squared_error(x_test1,ansReal))
temp_err3 =  mean_absolute_error(ans,ansReal)

print('Prediction error: ', temp_err)
print('Benchmark error: ',temp_err2)
print('Mean absolute error: ', temp_err3)

#Plotting of the results

plt.plot(epoch_count,train_loss)
plt.plot(epoch_count,val_loss)
plt.ylim((0,0.002))
plt.xlabel('Epoch')
plt.ylabel('Loss value')
plt.legend(['Training loss', 'Validation loss'])
plt.show()

aa=[x for x in range(len(ans[:,0]))]
plt.plot(aa, ans[:,0], marker='.', label="actual")
plt.plot(aa, ansReal[:,0], 'r', label="prediction")
plt.tight_layout()
plt.subplots_adjust(left=0.07)
plt.ylabel('Temperature', size=15)
plt.xlabel('Time step', size=15)
plt.legend(fontsize=15)
plt.show()