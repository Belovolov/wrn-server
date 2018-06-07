import cv2
import numpy as np
import wide_residual_network as wrn
from keras.datasets import cifar10
from keras import backend as K
import cifarMeta

img_rows, img_cols = 32, 32
(trainX, trainY), (testX, testY) = cifar10.load_data()
init_shape = (3, 32, 32) if K.image_dim_ordering() == 'th' else (32, 32, 3)

# For WRN-16-8 put N = 2, k = 8
model = wrn.create_wide_residual_network(init_shape, nb_classes=10, N=2, k=8, dropout=0.00)
model.load_weights("weights/WRN-16-8-Weights.h5")
print("Model loaded.")

def get_prediction(imagePath):
    image = cv2.imread(imagePath)
    
    arr = np.empty([1,32,32,3])
    arr[0] = image
    #that's the mean/std normalization
    trainX = np.concatenate(arr, trainX).astype('float32')
    trainX = (trainX - trainX.mean(axis=0)) / (trainX.std(axis=0))
    
    yPreds = model.predict(trainX[0:1]).round(2)
    yPred = np.argmax(yPreds, axis=1)
    #yPred = kutils.to_categorical(yPred)
    
    
    result = {}
    for i in range(0, len(yPreds[0])):
        if (yPreds[0][i]>0):
            result[cifarMeta.c10[i]]=yPreds[0][i]
    topResults = [(k, result[k]) for k in sorted(result, key=result.get, reverse=True)][0:5]
    
    return topResults