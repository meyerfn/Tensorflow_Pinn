import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
import os


def exactInitialCondition(x):
    return np.sin(x)


def createTrainingsamples(nSamples):
    nSamples = 2 * nSamples
    xSamples = np.linspace(0,
                           2. * np.pi,
                           nSamples,
                           endpoint=True,
                           dtype=np.float32)
    np.random.shuffle(xSamples)
    tSamples = np.linspace(0, 1.0, nSamples, endpoint=True)
    np.random.shuffle(tSamples)

    initialT = 0.
    xLeftBoundary = 0.
    xRightBoundary = 2. * np.pi

    initialConditionSamples = np.zeros(shape=(nSamples, 2))
    initialConditionSamples[:, 0] = initialT
    initialConditionSamples[:, 1] = xSamples

    boundaryConditionSamples = np.zeros(shape=(nSamples, 2))
    boundaryConditionSamples[:, 0] = tSamples
    boundaryConditionSamples[:, 1] = random.choices(
        [xLeftBoundary, xRightBoundary], k=nSamples)

    trueInitialcondition = np.zeros(shape=(nSamples, 1))
    trueInitialcondition[:,
                         0] = exactInitialCondition(initialConditionSamples[:,
                                                                            1])
    trueBoundaryData = np.zeros(shape=(nSamples, 1))
    trainingData = np.append(initialConditionSamples,
                             boundaryConditionSamples,
                             axis=0)
    referenceData = np.append(trueInitialcondition, trueBoundaryData, axis=0)

    return trainingData, referenceData


def createResidualTrainingsamples(nSamples):
    xSamples = np.linspace(0,
                           2. * np.pi,
                           nSamples,
                           endpoint=False,
                           dtype=np.float32)
    np.random.shuffle(xSamples)
    tSamples = np.linspace(0., 1.0, nSamples, endpoint=False)
    np.random.shuffle(tSamples)

    collocationPoints = np.zeros(shape=(nSamples, 2))

    collocationPoints[:, 0] = tSamples
    collocationPoints[:, 1] = xSamples

    residualTrainingData = collocationPoints
    return residualTrainingData


class Pinn(keras.Model):
    def __init__(self,
                 nLayers=5,
                 dimHidden=10,
                 dimIn=2,
                 dimOut=1,
                 batchSize=10,
                 nEpochs=100):
        super(Pinn, self).__init__()
        self.batchSize = batchSize
        self.nEpochs = nEpochs
        self.dimIn = dimIn
        self.dimOut = dimOut
        self.nLayers = nLayers
        self.dimHidden = dimHidden
        self.inputLayer = keras.Input(shape=(self.dimIn, ), batch_size=0)
        self.hiddenLayer = keras.layers.Dense(self.dimHidden,
                                              activation="tanh")
        self.outputLayer = keras.layers.Dense(self.dimOut, activation="linear")
        self.ffn = keras.Sequential(
            [
                self.inputLayer,
                keras.layers.Dense(self.dimHidden, activation="tanh"),
                keras.layers.Dense(self.dimHidden, activation="tanh"),
                self.outputLayer,
            ],
            name="test",
        )

    def compile(self, optimizer, loss_fn):
        super(Pinn, self).compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        self.ffn.trainable_weights
        with tf.GradientTape() as tape:
            predictions = self.ffn(x)
            loss = self.loss_fn(y, predictions)
        grads = tape.gradient(loss, self.ffn.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.ffn.trainable_weights))
        return {"loss": loss}

    def saveModel(self, path):
        self.ffn.save(path)


dimIn, dimHidden, dimOut, batchSize, nEpochs = 2, 10, 1, 100, 500
nSamples = 200
nSamplesResidual = 1000
trainingData, trueOutput = createTrainingsamples(nSamples)
residualTrainingData = createResidualTrainingsamples(nSamplesResidual)

# model = Pinn(dimIn, dimHidden, dimOut, batchSize, nEpochs)

model = Pinn()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss_fn=keras.losses.MeanSquaredError(),
)

model.fit(trainingData, trueOutput, epochs=nEpochs, batch_size=0)

# model.fit(trainingData, trueOutput,epochs=3, batch_size=2)
model.saveModel("{0}/pinn".format(os.getcwd()))
