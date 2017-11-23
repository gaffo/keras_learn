from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.models import load_model
import numpy as np

print("creating model")
model = Sequential()
model.add(Dense(3, activation='relu', input_dim=(3)))
model.add(Dense(9, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("creating trains")
x_train = []
y_train = []

for r in range(0, 255):
    for g in range(0, 255):
        for b in range(0, 255):
            v = (r, g, b)
            m = max(r, g, b)
            if m == r:
                c = 1
            elif m == g:
                c = 0
            elif m == b:
                c = 0

            x_train.append(v)
            y_train.append(c)

print("fitting")
model.fit(x_train, y_train, epochs=10, batch_size=4086)
print("fit")
model.summary()
score = model.evaluate(x_train, y_train, batch_size=4086)
print(score)
print(model.metrics_names)

model.save("colors.h5", True)

print(model.predict(x=np.array([(0, 3, 0)]), batch_size=1))
print(model.predict(x=np.array([(255, 0, 0)]), batch_size=1))