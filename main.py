import matplotlib.pyplot as plt
import numpy as np
import nnUtil

x, y = nnUtil.getSampleData()
# nnUtil.plotData(x, y)

no_of_neurons = 3

layers = 2

layer1 = np.random.randn(no_of_neurons, 2) + 0.5
layer2 = np.random.randn(no_of_neurons, 1) + 0.5
layer2_bias = np.random.randn(1)
print(layer1)
print(layer2)

epocs = 4
for epoc in range(epocs):
    y_pred = []
    x_valus = []
    for x_i in x:
        # forward prop
        o1 = np.zeros((no_of_neurons, 1))
        for i in range(no_of_neurons):
            o1[i][0] += layer1[i, 0] * x_i + layer1[i, 1]
            o1[i][0] = nnUtil.relu(o1[i][0])

        o2 = 0.0
        for i in range(no_of_neurons):
            o2 += layer2[i, 0] * o1[i, 0]

        y_i = o2 + layer2_bias
        y_i = nnUtil.relu(y_i)

        x_valus.append(x_i)
        y_pred.append(y_i)

    # plt.plot(x_valus, y_pred)
    # plt.grid()
    # plt.show()

    err_total = 0.0
    for i in range(len(y_pred)):
        x_i = x_valus[i]
        y_i = y_pred[i]
        err = (y_i - nnUtil.f(x_i))**2

        err_total += abs(err)

        # backprop

        # derr/derr = 1
        # derr/dy_i = (y_i > 0 ? 1 : 0)
        # derr/dlayer2_bias = dy_i/dlayer2_bias * derr/dy_i = (y_i > 0 ? 1 : 0) * 1
        # derr/dlayer2_weight_i =  dy_i/dlayer2_weight_i * derr/dy_i = (y_i > 0 ? 1 : 0) * layer1_neuron_i

        if y_i > 0:
            dy_i = 1 * (y_i - nnUtil.f(x_i))
        else:
            dy_i = 0

        lr = 0.01

        layer2_bias -= lr * dy_i

        for i in range(no_of_neurons):
            layer2[i, 0] -= lr * o1[i][0] * dy_i

    print("wieghts: ", layer1, layer2, layer2_bias)
    print("avg error = ", err/len(y_pred))


# print(nn)
