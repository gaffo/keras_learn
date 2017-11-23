from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import numpy as np

# create an early stopping callback which will stop us
# if the loss (error between runs) doesn't improve (go down) more than 0.001 over 100 epochs)
early_stopping = EarlyStopping(monitor='loss', mode='min', min_delta=0.00001, patience=100)

print("creating training sets")
input_train = [(0), (1)] # this means that we have 2 input classes, a 0, or a 1
output_train = [(0), (1)] # this means that when we have the first input (index alignment) we have a 0 out, and on the 2nd, a one out
# eg:
# 0 -> 0
# 1 -> 1
# basically this is the identity but it's slightly different
# as we're saying that 0 is not one, and 1 is one... becuase
# our output layer is binary... you can see this on the compile settings

# this solution doesn't seem to settle every time, this is because keras initalized models with random values in all of the nodes, so
# if it is going to work is luck of the draw
# so we run it 500 times or until it finds a fit
# generally when this doesn't settle it will go sideways with a accuracy of .5 and loss of 0.6931
# it also usually settles in a few runs so we're not really going to run it 500 times
for i in range(500):
    # we create the model in the loop as it's the easiest way to randomize it
    print("creating model")
    model = Sequential()
    model.add(Dense(1, activation='relu', input_dim=(1))) # create the first layer with one input (intput_dim) and one node (1).
    model.add(Dense(1, activation='relu')) # this number of nodes doesn't seem to matter
    model.add(Dense(1, activation='sigmoid')) # one output node

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    print("fitting")
    # fit the model over 10000 epochs (can take up to that long, small inputs so fast), with a batch size of 128 (seems to settle
    # on a correct solution more often than not, with the early stop monitor
    model.fit(input_train, output_train, epochs=10000, batch_size=4086, verbose=1, callbacks=[early_stopping])
    print("fit")
    score = model.evaluate(input_train, output_train, batch_size=4086)
    print(model.metrics_names)
    print(score)
    if score[1] > 0.99: # if accuracy is greater than .99, we have an accurate model
        print("Model settled into accuracy, we're done!")
        break
    print("Model went pathological so we're restarting")

model.save("identity.h5", overwrite=True) # it worked, so write the model out to a file. We can load it later with load_model (imported above)

# lets look at our actual values:
for input in [(0), (1)]:
    value = model.predict(x=np.array([input]), batch_size=1) # ask the model if it thinks it is a 0 or a 1
    print(input, value) # print the input and the classification of the output

# now lets look at what it thinks of non-integers
for input in np.arange(0, 0.99, 0.01):
    value = model.predict(x=np.array([input]), batch_size=1) # ask the model if it thinks it is a 0 or a 1
    print(input, value) # print the input and the classification of the output
