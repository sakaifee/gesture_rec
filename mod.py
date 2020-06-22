import numpy as np
import cv2
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from os import listdir
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

tags={"Hi":1,"Sir":2,"maam":3,"Morning":4,"Bye Thank You":5,"Good":6,"eating":7,"me":8}
##Enter Path
p="/Users/ar-rohan.sharma/Desktop/College_Proj/gesture-recognition/today/"
path=listdir(p)

##Get all the path of images in imagepaths
imagepaths=[]
for sub in path:
    if sub != ".DS_Store":
        subdir=p+sub+"/"
        a=listdir(subdir)
        for img in a:
            k=subdir+img
            imagepaths.append(k)

##Load images and labels
X = [] # Image data
Y = []  # Labels  
for images in imagepaths:
    image=cv2.imread(images, cv2.IMREAD_COLOR)
    image=np.array(image)
    X.append(image)
    category = images.split("/")[7]
    Y.append(tags[category])
X = np.array(X)
#X = np.stack((X,)*3, axis=1)
Y=np.array(Y)
print("Images loaded: ", len(X))
print("Labels loaded: ", len(Y))

ts = 0.3
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

# Construction of model
model = Sequential()
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(215, 240,3))) 
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu')) 
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
# Configures the model for training
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Trains the model for a given number of epochs (iterations on a dataset) and validates it.
model.fit(X_train, y_train, epochs=5, batch_size=19, verbose=2, validation_data=(X_test, y_test))
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save("model.h5")
print("Saved model to disk")
 