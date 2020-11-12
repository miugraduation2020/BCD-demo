from convert import convert
from preprocess import preprocess
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import keras
from keras.models import Sequential
from keras.layers import Dense

x = convert("train")
x_test = convert("test")

asl_train = pd.DataFrame(x)
X, y = preprocess(asl_train)

asl_test = pd.DataFrame(x_test)
X_test, y_test = preprocess(asl_test)

model = Sequential()
model.add(Dense(340, input_dim=X.shape[1], activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(y.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

history = model.fit(X, y, epochs=200, batch_size=190)

# Prediction phase
out = model.predict(X_test)
pred = list()
for i in range(len(out)):
    pred.append(np.argmax(out[i]))

test = list()
for i in range(len(y_test)):
    test.append(np.argmax(y_test[i]))

# Accuracy
a = accuracy_score(test, pred)
print(f'Accuracy: {round(a*100, 1)}%')

model.save('model.h5')
del model
