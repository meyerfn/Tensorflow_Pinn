from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np


def createPlotsamples(nSamples, t=0.):
    xSamples = np.linspace(-2., 2., nSamples, endpoint=True, dtype=np.float32)
    ySamples = np.linspace(-3.5, 0., nSamples, endpoint=True, dtype=np.float32)
    plotSamples = np.zeros(shape=(nSamples * nSamples, 3))
    cnt = 0
    for x in xSamples:
        for y in ySamples:
            plotSamples[cnt, 1] = x
            plotSamples[cnt, 2] = y
            cnt += 1
    plotSamples[:, 0] = t

    return plotSamples


model = keras.models.load_model("pinn")

plotData = createPlotsamples(1000, 0.)
y = model.predict(plotData)
plt.scatter(plotData[:, 1], plotData[:, 2], y[:, 0])
plt.show()
