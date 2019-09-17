import tensorflow as tf
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

#print(x_train[0])
#print(y_train[0])

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=x_train[0].shape))

#act = tf.keras.activations.relu(alpha = 0, max_value = 0.98, threshold = 0.01)

model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=4)
model.save('mnist_trained.h5')
model.summary()
new_model = tf.keras.models.load_model('mnist_trained.h5')

val_loss, val_acc = new_model.evaluate(x_test, y_test)
print(val_loss)
print(val_acc)

predictions = new_model.predict([x_test[0:10]])
#print(predictions)

import numpy as np
cleaned = np.zeros(10)
for i in range(10):
    cleaned[i] = np.argmax(predictions[i])

print(cleaned)
