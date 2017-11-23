from keras.models import load_model
import numpy as np

model = load_model("colors.h5")

for i in [(0, 255, 0), (0, 1, 0), (1, 0, 0), (2, 0, 0)]:
    print(i, model.predict(x=np.array([i]), batch_size=1))