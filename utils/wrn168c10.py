import cv2
import numpy as np
import wide_residual_network as wrn
from keras.datasets import cifar10
from keras import backend as K

img_rows, img_cols = 32, 32
(trainX, trainY), (testX, testY) = cifar10.load_data()
init_shape = (3, 32, 32) if K.image_dim_ordering() == 'th' else (32, 32, 3)

# For WRN-16-8 put N = 2, k = 8
model = wrn.create_wide_residual_network(init_shape, nb_classes=10, N=2, k=8, dropout=0.00)
model.load_weights("weights/WRN-16-8 Weights.h5")
print("Model loaded.")

def get_prediction(imagePath):
    image = cv2.imread(imagePath)
    
    arr = np.empty([1,32,32,3])
    arr[0] = image
    #that's the mean/std normalization
    trainX = np.concatenate(arr, trainX).astype('float32')
    trainX = (trainX - trainX.mean(axis=0)) / (trainX.std(axis=0))
    
    yPreds = model.predict(trainX[0:1])
    yPred = np.argmax(yPreds, axis=1)
    #yPred = kutils.to_categorical(yPred)
    
    return yPred[0]
    # print(yPreds)
    # print(yPred)
    # print(trainY[0:20].flatten())
    #print (testY - testX[3:100])
    
    #yPreds2 = model.predict(arr2)
    #yPred2 = np.argmax(yPreds2, axis=1)
    #yPred2 = kutils.to_categorical(yPred2)
    #print(yPreds2)
    #print(yPred2)