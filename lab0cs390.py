import urllib.request
import requests
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.metrics import classification_report


# Setting random seeds to keep everything deterministic.
random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

# Disable some troublesome logging.
#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Information on dataset.
NUM_CLASSES = 10
IMAGE_SIZE = 784

# Use these to set the algorithm to use.
#ALGORITHM = "guesser"
#ALGORITHM = "custom_net"
ALGORITHM = "tf_net"





class NeuralNetwork_2Layer():
    def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate = 0.1):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
        self.W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)

    # Activation function.
    def __sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    # Activation prime function.
    def __sigmoidDerivative(self, x):
        value = self.__sigmoid(x)
        return value*(1-value)


    # Batch generator for mini-batches. Not randomized.
    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]

    def __batchGenerator2(self, l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]

    # Training with backpropagation.
    def train(self, xVals, yVals, epochs = 100000, minibatches = True, mbs = 100):
        orig_x, orig_y = xVals, yVals
        for ind in range(200):
          #xVals, yVals = next(self.__batchGenerator(orig_x, mbs)), next(self.__batchGenerator2(orig_y, mbs))
          layer1, layer2 = self.__forward(xVals)
          change_weights_2 = np.dot(layer1.T, (2*(yVals - layer2) * self.__sigmoidDerivative(layer2)))
          change_weights_1 = np.dot(xVals.T, (np.dot(2*(yVals - layer2) * self.__sigmoidDerivative(layer2), self.W2.T) * self.__sigmoidDerivative(layer1)))                                   #TODO: Implement backprop. allow minibatches. mbs should specify the size of each minibatch.
          print(ind)
          self.W1 += self.lr * change_weights_1
          self.W2 += self.lr * change_weights_2

    # Forward pass.
    def __forward(self, input):
        layer1 = self.__sigmoid(np.dot(input, self.W1))
        layer2 = self.__sigmoid(np.dot(layer1, self.W2))
        return layer1, layer2

    # Predict.
    def predict(self, xVals):
        _, layer2 = self.__forward(xVals)
        return layer2



# Classifier that just guesses the class label.
def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)



#=========================<Pipeline Functions>==================================

def getRawData():
    """
    try:
        urllib.request.urlopen("http://google.com")
        alpha = True
    except:
        alpha = False
    print(alpha)
    """
    #mnist = requests.get("https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz", verify=False)
    mnist = tf.keras.datasets.mnist
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))



def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw            #TODO: Add range reduction here (0-255 ==> 0.0-1.0).
    xTrain = np.reshape(xTrain, (60000, 784))
    xTrain = xTrain.astype('float32')/255.0
    xTest = np.reshape(xTest, (10000, 784))
    xTest = xTest.astype('float32')/255.0
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrain, yTrainP), (xTest, yTestP))



def trainModel(data):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "custom_net":
        print("Building and training Custom_NN.")
        model = NeuralNetwork_2Layer(784, 10, 130)
        model.train(xTrain, yTrain)
        return model

        #print("Not yet implemented.")                   #TODO: Write code to build and train your custon neural net.
        #return None
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        return trainModelKeras(data)
        #print("Not yet implemented.")                   #TODO: Write code to build and train your keras neural net.
        #return None
    else:
        raise ValueError("Algorithm not recognized.")



def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "custom_net":
        print("Testing Custom_NN.")
        return testModel(data, model)
        #print("Not yet implemented.")                   #TODO: Write code to run your custon neural net.
        #return None
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        return testModel(data, model)
        #print("Not yet implemented.")                   #TODO: Write code to run your keras neural net.
        #return None
    else:
        raise ValueError("Algorithm not recognized.")



def evalResults(data, preds):   #TODO: Add F1 score confusion matrix here.
    xTest, yTest = data
    acc = 0
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
    accuracy = acc / preds.shape[0]
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print("F1 score:")
    print(classification_report(yTest, preds))

def trainModelKeras(data):
    model = Sequential()
    model.add(Dense(10, kernel_initializer='random_normal', input_dim=(IMAGE_SIZE), activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(10, kernel_initializer='random_normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(data[0], data[1], epochs=30, batch_size=64)
    return model

def testModel(data, model):
    test_results = model.predict(data)
    #print(test_results[0])
    #print(test_results[0].shape)
    for j in range(test_results.shape[0]):
    #for j in range(1):
      max_index, max_value = -1, 0
      for i in range(10):
        if test_results[j][i] > max_value:
          max_index, max_value = i, test_results[j][i]
        #print(test_results[j][i], max_value, max_index)
      test_results[j] = np.array([0] * 10)
      test_results[j][max_index] = 1
      #print(test_results[j])
    return test_results

#=========================<Main>================================================

def main():
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)



if __name__ == '__main__':
    main()
