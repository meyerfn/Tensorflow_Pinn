import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
import os


def exactInitialCondition(x):
    # return np.exp(-( x[:, 0]**2 + x[:, 1]**2 )  / (2 * 0.2**2) )
    return x[:, 0]*(1.-x[:, 0])*x[:, 1]*(1. - x[:, 1])


def createTrainingDataInitialCondition(nSamples):
    initialConditionSamples = np.zeros(shape=(nSamples**2, 3))
    initialConditionSamples[:, 0] = 0.
    initialConditionSamples[:, 1] = np.random.uniform(
        low=0., high=1.0, size=(nSamples**2,))
    initialConditionSamples[:, 2] = np.random.uniform(
        low=0., high=1.0, size=(nSamples**2,))

    trueInitialcondition = np.zeros(shape=(nSamples**2, 1))
    trueInitialcondition[:, 0] = exactInitialCondition(
        initialConditionSamples[:, 1:3])
    return initialConditionSamples, trueInitialcondition


def createTrainingDataTimeDerivativeInitialCondition(nSamples):
    initialT = 0.
    timeDerivativeInitialConditionSamples = np.zeros(shape=(nSamples**2, 3))
    timeDerivativeInitialConditionSamples[:, 0] = initialT
    timeDerivativeInitialConditionSamples[:, 1] = np.random.uniform(
        low=0., high=1.0, size=(nSamples**2,))
    timeDerivativeInitialConditionSamples[:, 2] = np.random.uniform(
        low=0., high=1.0, size=(nSamples**2,))

    trueTimeDerivativeInitialcondition = np.zeros(shape=(nSamples**2, 1))

    return timeDerivativeInitialConditionSamples, trueTimeDerivativeInitialcondition


def createTrainingDataBoundaryCondition(nSamples):
    xLeftBoundary = 0.
    xRightBoundary = 1.
    yLeftBoundary = 0.
    yRightBoundary = 1.
    tSamples = np.random.uniform(low=0., high=1.0, size=(nSamples**2,))

    boundaryConditionSamples = np.zeros(shape=(nSamples**2, 3))
    boundaryConditionSamples[:, 0] = tSamples
    boundaryConditionSamples[:, 1] = random.choices(
        [xLeftBoundary, xRightBoundary], k=nSamples**2)
    boundaryConditionSamples[:, 2] = random.choices(
        [yLeftBoundary, yRightBoundary], k=nSamples**2)

    trueBoundaryData = np.zeros(shape=(nSamples**2, 1))

    return boundaryConditionSamples, trueBoundaryData


def createTrainingDataResidual(nSamples):
    residualSamples = np.zeros(shape=(nSamples**2, 3))
    residualSamples[:, 0] = np.random.uniform(
        low=0., high=1.0, size=(nSamples**2,))
    residualSamples[:, 1] = np.random.uniform(
        low=0., high=1.0, size=(nSamples**2,))
    residualSamples[:, 2] = np.random.uniform(
        low=0., high=1.0, size=(nSamples**2,))

    trueResidual = np.zeros(shape=(nSamples**2, 1))

    return residualSamples, trueResidual


class Pinn(keras.Model):
    def __init__(self, dimIn=3, dimHidden=5, dimOut=1, nSamples=100):
        super(Pinn, self).__init__()
        self.batchSize = batchSize
        self.nEpochs = nEpochs
        self.dimIn = dimIn
        self.dimOut = dimOut
        self.nSamples = nSamples
        self.dimHidden = dimHidden
        self.inputLayer = keras.Input(shape=(self.dimIn, ))
        self.outputLayer = keras.layers.Dense(self.dimOut, activation="linear")
        self.ffn = keras.Sequential(
            [
                self.inputLayer,
                keras.layers.Dense(self.dimHidden, activation="tanh"),
                keras.layers.Dense(self.dimHidden, activation="tanh"),
                keras.layers.Dense(self.dimHidden, activation="tanh"),
                keras.layers.Dense(self.dimHidden, activation="tanh"),
                keras.layers.Dense(self.dimHidden, activation="tanh"),
                self.outputLayer
            ],
            name="ffn",
        )

    def compile(self, optimizer, loss_fn):
        super(Pinn, self).compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        dataInitialCondition, trueInitial = data[0][0], data[1][0]
        dataTimeDerivativeInitialCondition, trueDtInitial = data[0][1], data[
            1][1]
        tDerivative = dataTimeDerivativeInitialCondition[:, 0]
        xDerivative = dataTimeDerivativeInitialCondition[:, 1]
        yDerivative = dataTimeDerivativeInitialCondition[:, 2]
        dataBoundaryCondition, trueBoundary = data[0][2], data[1][2]
        dataResidual, trueResidual = data[0][3], data[1][3]
        tResidual = dataResidual[:, 0]
        xResidual = dataResidual[:, 1]
        yResidual = dataResidual[:, 2]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(dataInitialCondition)
            tape.watch(dataBoundaryCondition)
            tape.watch(tResidual)
            tape.watch(xResidual)
            tape.watch(yResidual)
            tape.watch(tDerivative)
            tape.watch(xDerivative)
            tape.watch(yDerivative)

            u = self.ffn(tf.stack([tResidual, xResidual, yResidual], 1))
            dtU = tape.gradient(u, tResidual)
            dtdtU = tape.gradient(dtU, tResidual)
            dxU = tape.gradient(u, xResidual)
            dxdxU = tape.gradient(dxU, xResidual)
            dyU = tape.gradient(u, yResidual)
            dydyU = tape.gradient(dyU, yResidual)
            residual = dtdtU + dxdxU + dydyU

            uInitial = self.ffn(
                tf.stack([tDerivative, xDerivative, yDerivative], 1))
            dtUInitial = tape.gradient(uInitial, tDerivative)

            predictionInitial = self.ffn(dataInitialCondition)
            predictionBoundaryCondition = self.ffn(
                dataBoundaryCondition)

            loss = self.loss_fn(trueInitial, predictionInitial)
            loss += self.loss_fn(trueResidual, residual)
            loss += self.loss_fn(trueDtInitial, dtUInitial)
            loss += self.loss_fn(trueBoundary[0:nSamples, :],
                                 predictionBoundaryCondition[0:nSamples, :])

        grads = tape.gradient(loss, self.ffn.trainable_weights)

        self.optimizer.apply_gradients(zip(grads, self.ffn.trainable_weights))
        del tape
        return {"loss": loss}

    def saveModel(self, path):
        self.ffn.save(path)


dimIn, dimHidden, dimOut, batchSize, nEpochs = 3, 5, 1, 20, 100
nSamples = 1000

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

# tf.config.run_functions_eagerly(True)
model = Pinn(dimIn, dimHidden, dimOut)
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss_fn=keras.losses.MeanSquaredError(),
)
model.fit(trainingList,
          trueDataList,
          epochs=nEpochs,
          batch_size=nSamples,
          shuffle=True)

model.saveModel("{0}/pinn".format(os.getcwd()))
