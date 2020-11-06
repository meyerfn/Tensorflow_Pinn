import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
import os


def exactInitialCondition(x):
    return np.exp(-( (x[:, 0] - 0)**2 + (x[:, 1] - (0) )**2) / (2 * 0.2**2))


def createTrainingDataInitialCondition(nSamples):
    initialT = 0.
    xSamples = np.linspace(-2., 2., nSamples, endpoint=True)
    ySamples = np.linspace(-2., 2., nSamples, endpoint=True)
    np.random.shuffle(xSamples)
    np.random.shuffle(ySamples)

    initialConditionSamples = np.zeros(shape=(nSamples**2, 3))
    initialConditionSamples[:, 0] = initialT
    cnt = 0
    for x in xSamples:
        for y in ySamples:
            initialConditionSamples[cnt, 1] = x
            initialConditionSamples[cnt, 2] = y
            cnt += 1

    trueInitialcondition = np.zeros(shape=(nSamples**2, 1))
    trueInitialcondition[:, 0] = exactInitialCondition(
        initialConditionSamples[:, 1:3])

    return initialConditionSamples, trueInitialcondition


def createTrainingDataTimeDerivativeInitialCondition(nSamples):
    initialT = 0.
    xSamples = np.linspace(-2., 2., nSamples**2, endpoint=True)
    ySamples = np.linspace(-2., 2., nSamples**2, endpoint=True)
    np.random.shuffle(xSamples)
    np.random.shuffle(ySamples)

    timeDerivativeInitialConditionSamples = np.zeros(shape=(nSamples**2, 3))
    timeDerivativeInitialConditionSamples[:, 0] = initialT
    timeDerivativeInitialConditionSamples[:, 1] = xSamples
    timeDerivativeInitialConditionSamples[:, 2] = ySamples

    trueTimeDerivativeInitialcondition = np.zeros(shape=(nSamples**2, 1))

    return timeDerivativeInitialConditionSamples, trueTimeDerivativeInitialcondition


def createTrainingDataBoundaryCondition(nSamples):
    xLeftBoundary = -2.
    xRightBoundary = 2.
    yLeftBoundary = -2
    yRightBoundary = 2.
    tSamples = np.linspace(0, 1.0, nSamples**2, endpoint=True)
    np.random.shuffle(tSamples)

    boundaryConditionSamples = np.zeros(shape=(nSamples**2, 3))
    boundaryConditionSamples[:, 0] = tSamples
    boundaryConditionSamples[:, 1] = random.choices(
        [xLeftBoundary, xRightBoundary], k=nSamples**2)
    boundaryConditionSamples[:, 2] = random.choices(
        [yLeftBoundary, yRightBoundary], k=nSamples**2)

    trueBoundaryData = np.zeros(shape=(nSamples**2, 1))

    return boundaryConditionSamples, trueBoundaryData


def createTrainingDataResidual(nSamples):
    xSamples = np.linspace(-2., 2., nSamples**2, endpoint=False)
    ySamples = np.linspace(-2., 2., nSamples**2, endpoint=False)
    np.random.shuffle(xSamples)
    np.random.shuffle(ySamples)
    tSamples = np.linspace(0., 1.0, nSamples**2, endpoint=False)
    np.random.shuffle(tSamples)

    residualSamples = np.zeros(shape=(nSamples**2, 3))
    residualSamples[:, 0] = tSamples
    residualSamples[:, 1] = xSamples
    residualSamples[:, 2] = ySamples
    trueResidual = np.zeros(shape=(nSamples**2, 1))

    return residualSamples, trueResidual


class Pinn(keras.Model):
    def __init__(self, dimHidden=5, dimIn=3, dimOut=1, nEpochs=100):
        super(Pinn, self).__init__()
        self.batchSize = batchSize
        self.nEpochs = nEpochs
        self.dimIn = dimIn
        self.dimOut = dimOut
        self.dimHidden = dimHidden
        self.inputLayer = keras.Input(shape=(self.dimIn, ))
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
            name="ffn",
        )

    def compile(self, optimizer, loss_fn):
        super(Pinn, self).compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        dataInitialCondition, yInitial = data[0][0], data[1][0]
        dataTimeDerivativeInitialCondition, yTimeDerivativeInitial = data[0][
            1], data[1][1]
        dataBoundaryCondition, yBoundary = data[0][2], data[1][2]
        dataResidual, yResidual = data[0][3], data[1][3]

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(dataResidual)
            tape.watch(dataTimeDerivativeInitialCondition)

            predictionInitialCondition = self.ffn(dataInitialCondition)
            predictionBoundaryCondition = self.ffn(dataBoundaryCondition)

            uInitial = self.ffn(dataTimeDerivativeInitialCondition)
            u = self.ffn(dataResidual)

            gradientUInitual = tape.gradient(
                uInitial, dataTimeDerivativeInitialCondition)
            gradientU = tape.gradient(u, dataResidual)
            secondOrderGradientU = tape.gradient(gradientU, dataResidual)

            loss = self.loss_fn(yInitial, predictionInitialCondition)
            # loss += self.loss_fn(yTimeDerivativeInitial, gradientUInitual[:,
            #                                                               0])
            # loss += self.loss_fn(yBoundary, predictionBoundaryCondition)
            # loss += self.loss_fn(
            #     yResidual, secondOrderGradientU[:, 0] +
            #     secondOrderGradientU[:, 1] + secondOrderGradientU[:, 2])

        grads = tape.gradient(loss, self.ffn.trainable_weights)

        self.optimizer.apply_gradients(zip(grads, self.ffn.trainable_weights))
        del tape
        return {"loss": loss}

    def saveModel(self, path):
        self.ffn.save(path)


dimIn, dimHidden, dimOut, batchSize, nEpochs = 2, 20, 1, 100, 100
nSamples = 100
nSamplesResidual = 100

trainingDataInitialCondition, trueInitialData = createTrainingDataInitialCondition(
    nSamples)
trainingDataTimeDerivativeInitialCondition, trueTimeDerivativeInitialData = createTrainingDataTimeDerivativeInitialCondition(
    nSamples)
trainingDataBoundaryCondition, trueBoundaryData = createTrainingDataBoundaryCondition(
    nSamples)
trainingDataResidual, trueResidual = createTrainingDataResidual(nSamples)

trainingList = []
trainingList.append(trainingDataInitialCondition)
trainingList.append(trainingDataTimeDerivativeInitialCondition)
trainingList.append(trainingDataBoundaryCondition)
trainingList.append(trainingDataResidual)

trueDataList = []
trueDataList.append(trueInitialData)
trueDataList.append(trueTimeDerivativeInitialData)
trueDataList.append(trueBoundaryData)
trueDataList.append(trueResidual)

tf.config.run_functions_eagerly(True)
model = Pinn()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.005),
    loss_fn=keras.losses.MeanSquaredError(),
)
model.fit(trainingList, trueDataList, epochs=nEpochs, batch_size=500)

model.saveModel("{0}/pinn".format(os.getcwd()))
