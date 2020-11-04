from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

def createPlotsamples(spatialRefinement, tEval=0.):
    xSamples = np.linspace(0,
                           2. * np.pi,
                           spatialRefinement,
                           endpoint=True,
                           dtype=np.float32)
    plotSamples = np.zeros(shape=(spatialRefinement, 2), dtype=np.float32)
    plotSamples[:, 0] = tEval
    plotSamples[:, 1] = xSamples
    return plotSamples

def exactFunction(x):
    return np.sin(x)

model = keras.models.load_model("pinn")

plotData = createPlotsamples(1000)
y = model.predict(plotData)
yExact= exactFunction(plotData[:,1])

plt.plot(plotData[:, 1], y, label= 'NN approximation, t=0.')
plt.plot(plotData[:, 1], yExact, label= 'Exact solution, t=0.')

plt.legend()
plt.show()



