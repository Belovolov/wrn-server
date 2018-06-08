import cv2
import numpy as np
import wide_residual_network as wrn
from keras.datasets import cifar10
from keras import backend as K
import tensorflow as tf
import cifarMeta

class wrn168c10():
    def __init__(self, weightsFile):
        (self.trainX, trainY), (testX, testY) = cifar10.load_data()
        init_shape = (3, 32, 32) if K.image_dim_ordering() == 'th' else (32, 32, 3)
        self.model = wrn.create_wide_residual_network(init_shape, nb_classes=10, N=2, k=8, dropout=0.00)
        self.model.load_weights(weightsFile)
        self.graph = tf.get_default_graph()
    def get_prediction(self, imageData):
        img = np.frombuffer(imageData, np.uint8)
        image = cv2.imdecode(img, cv2.IMREAD_COLOR).astype('float32')
        
        arr = np.empty([1,32,32,3])
        arr[0] = image
        
        #that's the mean/std normalization
        trainX_t = np.concatenate((arr, self.trainX.astype('float32')))
        print(trainX_t[0:1])
        trainX_t = (trainX_t - trainX_t.mean(axis=0)) / (trainX_t.std(axis=0))
        print(trainX_t[0:1])
        
        with self.graph.as_default():
            yPreds = self.model.predict(trainX_t[0:1]).round(2)
        
        result = {}
        for i in range(0, len(yPreds[0])):
            if (yPreds[0][i]>0):
                result[cifarMeta.c10[i]]=str(yPreds[0][i])
        topResults = [(k, result[k]) for k in sorted(result, key=result.get, reverse=True)]
        print(topResults)
        
        return result