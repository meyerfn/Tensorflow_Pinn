from tensorflow import keras
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from mpl_toolkits.mplot3d.axes3d import get_test_data
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

def exactInitialCondition(x):
    # return np.exp(-((x[:, 0] - 0)**2 + (x[:, 1] - (0))**2) / (2 * 0.2**2))
    return x[:, 0]*(1.-x[:, 0])*x[:, 1]*(1. - x[:, 1])


def createPlotsamples(nSamples, t):
    xSamples = np.linspace(0., 1., nSamples, endpoint=True)
    ySamples = np.linspace(0., 1., nSamples, endpoint=True)
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
plotData = createPlotsamples(200, 0.)
y = model.predict(plotData)
yExact = exactInitialCondition(plotData[:, 1:3])
# yErr = (y[:,0]-yExact[:])

fig = plt.figure(figsize=plt.figaspect(0.5))
# ax = fig.add_subplot(1, 1, 1, projection='3d')
ax = fig.add_subplot(1, 1, 1, projection='3d')

cb = ax.scatter(plotData[:, 1],
           plotData[:, 2],
           y,
           c=y,
           s=5,
           marker='o',
           cmap=cm.viridis,
           )
plt.colorbar(cb)

# ax = fig.add_subplot(1, 2, 2, projection='3d')
# cb2 = ax.scatter(plotData[:, 1],
#            plotData[:, 2],
#            yErr,
#            c=yErr,
#            s=5,
#            marker='o',
#            cmap=cm.viridis)
# plt.colorbar(cb2)
plt.show()