import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

training_set = pd.read_csv('TSet6-9.csv')
training_setnp = training_set.to_numpy()

test_index = 500

print(training_setnp.shape)

x_traintf = training_setnp[0:test_index,0:54]
y_traintf = training_setnp[0:test_index,54:72]

print(y_traintf.shape)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(54, input_dim = 54, activation='linear'))
#model.add(tf.keras.layers.Dense(50, activation='relu'))
model.add(tf.keras.layers.Dense(250,activation='linear',kernel_regularizer=tf.keras.regularizers.l2(0.001)))
#model.add(tf.keras.layers.Dense(100, activation='linear'))
model.add(tf.keras.layers.Dense(18,kernel_regularizer=tf.keras.regularizers.l2(0.001)))

model.compile(optimizer='adam',
              loss='mean_squared_error')

model.fit(x=x_traintf,y=y_traintf, epochs=250,shuffle=True)

val_loss = model.evaluate(x=training_setnp[test_index:training_setnp.shape[0],0:54],y=training_setnp[test_index:training_setnp.shape[0],54:72])
ans = model.predict([training_setnp[test_index:training_setnp.shape[0],0:54]])

print(val_loss)

temp_err = np.sqrt((0.01/18*((ans-ansReal)**2).values.sum())) #calculate the temperature error for the model

print(temp_err)

plt.plot(val_loss)
plt.show();