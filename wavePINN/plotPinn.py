from tensorflow import keras
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


def createPlotsamples(nSamples, t=0.):
    xSamples = np.linspace(-5., 5., nSamples, endpoint=True, dtype=np.float32)
    ySamples = np.linspace(-5, 5., nSamples, endpoint=True, dtype=np.float32)
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

plotData = createPlotsamples(120, 0.)
y = model.predict(plotData)
fig = plt.figure()
ax = fig.add_subplot()
cb = ax.scatter(plotData[:, 1],
           plotData[:, 2],
           vmin=-0.1,
           vmax=.1,
           s=35,
           c=y,
           marker='o',
           cmap=cm.jet)
plt.colorbar(cb)
plt.show()