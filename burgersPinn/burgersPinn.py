import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
import os


def exactInitialCondition(x):
    return np.sin(x)


def createTrainingDataInitialCondition(nSamples):
    initialT = 0.
    xSamples = np.linspace(0,
                           2. * np.pi,
                           nSamples,
                           endpoint=True,
                           dtype=np.float32)
    np.random.shuffle(xSamples)

    initialConditionSamples = np.zeros(shape=(nSamples, 2))
    initialConditionSamples[:, 0] = initialT
    initialConditionSamples[:, 1] = xSamples

    trueInitialcondition = np.zeros(shape=(nSamples, 1))
    trueInitialcondition[:,
                         0] = exactInitialCondition(initialConditionSamples[:,
                                                                            1])

    return initialConditionSamples, trueInitialcondition


def createTrainingDataBoundaryCondition(nSamples):
    xLeftBoundary = 0.
    xRightBoundary = 2. * np.pi
    tSamples = np.linspace(0, 1.0, nSamples, endpoint=True)
    np.random.shuffle(tSamples)

    boundaryConditionSamples = np.zeros(shape=(nSamples, 2))
    boundaryConditionSamples[:, 0] = tSamples
    boundaryConditionSamples[:, 1] = random.choices(
        [xLeftBoundary, xRightBoundary], k=nSamples)

    trueBoundaryData = np.zeros(shape=(nSamples, 1))

    return boundaryConditionSamples, trueBoundaryData


def createResidualTrainingsamples(nSamples):
    xSamples = np.linspace(0,
                           2. * np.pi,
                           nSamples,
                           endpoint=False,
                           dtype=np.float32)
    np.random.shuffle(xSamples)
    tSamples = np.linspace(0., 1.0, nSamples, endpoint=False)
    np.random.shuffle(tSamples)

    residualSamples = np.zeros(shape=(nSamples, 2))
    residualSamples[:, 0] = tSamples
    residualSamples[:, 1] = xSamples
    trueResidual = np.zeros(shape=(nSamples, 1))

    return residualSamples, trueResidual


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
        xInitial, yInitial = data[0][0], data[1][0]
        xBoundary, yBoundary = data[0][1], data[1][1]
        xResidual, yResidual = data[0][2], data[1][2]

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(xResidual)
            predictions = self.ffn(xInitial)
            loss = self.loss_fn(yInitial, predictions)
            predictions = self.ffn(xBoundary)
            loss += self.loss_fn(yBoundary, predictions)
            u = self.ffn(xResidual)
            gradientU = tape.gradient(u, xResidual)
            fluxU = 0.5 * u * u
            gradientFluxU = tape.gradient(fluxU, xResidual)
            loss += self.loss_fn(yResidual, gradientU[:,0]+gradientFluxU[:,1] )

        grads = tape.gradient(loss, self.ffn.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.ffn.trainable_weights))
        # del tape
        return {"loss": loss}

    def saveModel(self, path):
        self.ffn.save(path)


dimIn, dimHidden, dimOut, batchSize, nEpochs = 2, 10, 1, 100, 500
nSamples = 200
nSamplesResidual = 200

trainingDataInitialCondition, trueInitialData = createTrainingDataInitialCondition(
    nSamples)
trainingDataBoundaryCondition, trueBoundaryData = createTrainingDataBoundaryCondition(
    nSamples)
trainingDataResidual, trueResidual = createResidualTrainingsamples(
    nSamplesResidual)

trainingList = []
trainingList.append(trainingDataInitialCondition)
trainingList.append(trainingDataBoundaryCondition)
trainingList.append(trainingDataResidual)

trueDataList = []
trueDataList.append(trueInitialData)
trueDataList.append(trueBoundaryData)
trueDataList.append(trueResidual)

tf.config.run_functions_eagerly(True)
model = Pinn()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss_fn=keras.losses.MeanSquaredError(),
)
model.fit(trainingList, trueDataList, epochs=nEpochs, batch_size=10,shuffle=True)

model.saveModel("{0}/pinn".format(os.getcwd()))
