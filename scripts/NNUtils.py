import numpy as np
import math

class Neuron:
    def __init__(self, num_inputs, activation_fun):
        self.num_inputs = num_inputs
        self.activation_fun = activation_fun
        self.weights = np.random.rand(num_inputs + 1)

    def output(self, inputs):
        sum = 1 * self.weights[0]

        for i in range(0, self.num_inputs):
            sum += self.weights[i + 1] * self.inputs()

        return self.ReLU(sum)

    def ReLU(x):
        retval = 0
        if x > 0:
            retval = x

        return retval

    def ReLU_derivative(x):
        retval = 0
            if x > 0:
                retval = 1

        return retval

class Layer:
    def __init__(self, num_inputs, num_neurons):
        self.neurons = []

        for i in range(0, num_neurons):
            self.neurons.append()









def train():
    pass

def load_data():
    pass

def feed_forward():
    pass

def evaluate_network():
    pass

def backpropogation():
    pass

def onehot_to_class(output_array, rows, cols):
    """
    Takes an input like:
    100
    010
    001
    100
    001
    and ouputs: [1 2 3 1 3]


    :return:
    :rtype:
    """
    #naive approach for c implementation

    out = [] # need an out buffer in c impl
    for row in range(0,rows): # for (int row = 0; row < rows; i++)
        for col in range(0,cols): # for (int col = 0; col < cols; col++)
            if col == 1:
                out.append(col) #
                break

    return out

def class_to_onehot(class_array , len, num_classes):
    """
    takes an input like [1 2 3 1 3] and turns it into
    100
    010
    001
    100
    001
    :param class_array:
    :type class_array:
    :return:
    :rtype:
    """

    #naive approach for c impl.
    out = [] # len*classes sized buffer
    for i in range(0,len): # for (int i = 0; i < len; i++)
        temp = [0] * num_classes
        temp[class_array[i] - 1] = 1
        out.append(temp)

    return out

def sigmoid(x):
    return 1 / (1 + math.exp(x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))





