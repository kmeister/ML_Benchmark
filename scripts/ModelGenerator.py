#!/usr/bin/env python3
"""
MIT License

Copyright (c) 2020 Kurt Lee Meister Jr.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# generate header file containing weights and test data inputs for a 3 layer MLP
# read the code to figure out the hyperparams


from sklearn.datasets import make_blobs
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from matplotlib import pyplot
import numpy as np
import datetime
import argparse


def generate_data(n_samples=10000, n_classes=2, n_features=3, test_ratio=0.25):
    # generate 2d classification dataset
    X, y = make_classification(n_samples=n_samples, n_classes=n_classes, n_features=n_features, n_informative=n_features, n_redundant=0 , random_state=2)
    # one hot encode output variable
    y = to_categorical(y)
    # split into train and test
    trainX, testX, trainy, testy = train_test_split(X,y, test_size=test_ratio, random_state=42 )

    return trainX, testX, trainy, testy


def generate_model(trainX, trainy, input_size=3, layer_1_neurons = 10, layer_2_neurons =5, layer_3_neurons = 2, verbose = 1, n_epochs = 100):
    # define the layers
    layer_1 = Dense(layer_1_neurons, activation='relu', kernel_initializer=keras.initializers.he_uniform(seed=42), input_shape=(input_size,))
    layer_2 = Dense(layer_2_neurons, activation='relu', kernel_initializer=keras.initializers.he_uniform(seed=43))
    layer_3 = Dense(layer_3_neurons, activation='softmax', kernel_initializer=keras.initializers.glorot_uniform(seed=44))

    # define model

    model = Sequential(layers=[layer_1, layer_2, layer_3])

    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    # fit model
    model.fit(trainX, trainy, epochs=n_epochs, verbose=verbose)

    if verbose:
        model.summary()

    return model

def evaluate_model(model, trainX, testX, trainy, testy):
    # evaluate the model
    _, train_acc = model.evaluate(trainX, trainy, verbose=0)
    _, test_acc = model.evaluate(testX, testy, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))



def weights_to_c_array_declaration(layer, name):
    weights = layer.get_weights()[0].transpose()
    bias_waight = layer.get_weights()[1]
    all_weights = np.column_stack((bias_waight, weights))

    retval = f"// Dimensions and Weights (including biases in first col) for {name}\n"
    retval += f"const int {name}_N_NEURONS = {all_weights.shape[0]};\n"
    retval += f"const int {name}_N_WEIGHTS = {all_weights.shape[1]};\n"

    retval += f"float {name}_WEIGHTS[{all_weights.shape[0]}*{all_weights.shape[1]}] = " + "{\n"

    for neuron in range(0,all_weights.shape[0]):
        row = "\t "
        for w in range(0, all_weights.shape[1]):
            row += f"{all_weights[neuron][w]} "

            if w != all_weights.shape[1]-1:
                row += ", "

        if neuron == all_weights.shape[0] - 1:
            row +="\n"
        else:
            row +=",\n"

        retval += row

    retval += "};\n\n"

    return retval

def inputs_to_c_array(inputs, name):
    n_inputs = inputs.shape[0]
    n_features = inputs.shape[1]
    retval = f"// Array containing {name}  input values for the ANN\n"
    retval += f"const int {name}_N_INPUTS = {n_inputs};\n"
    retval += f"const int {name}_N_FEATURES = {n_features};\n"
    retval += f"float {name}_INPUTS[{n_inputs}*{n_features}] = " + "{\n\t"

    for i, input in enumerate(inputs):
        for j, feature in enumerate(input):

            retval += f"{feature} "

            if j != n_features -1:
                retval += ", "

        if i == n_inputs -1:
            retval += "\n};\n\n"
        else:
            retval += ",\n\t"

    return retval


def write_header(inputs, model, filename, header_guard="BENCHMARKS_WEIGHTS_H"):


    with open(filename, 'w') as file:
        file.write("//\n// autogenerated weights for a 3 hidden layer MLP\n")
        file.write(f"// Model Generator Created by Kurt Meister, Model generated  {datetime.datetime.now()}\n//\n\n")
        file.write(f"#ifndef {header_guard}\n#define {header_guard}")

        for i, layer in enumerate(model.layers):
            file.write(weights_to_c_array_declaration(layer,f"LAYER_{i+1}"))

        file.write(inputs_to_c_array(inputs, "TEST"))

        file.write(f"#endif //{header_guard}")

def dump_all(model,inputs):
    for i, layer in enumerate(model.layers):
        print(weights_to_c_array_declaration(layer, f"LAYER_{i}"))

    print(inputs_to_c_array(inputs, "TEST"))

    print("Dumping Sample Data:\n\n")

    input = inputs[0:1]

    print(f"Sample Input: {input}\n")
    print(f"Model Prediction: {model.predict(input)}")

    for i, layer in enumerate(model.layers):
        layermodel = keras.Model(inputs=model.input, outputs=layer.output)
        print(f"Layer {i+1} Output: {layermodel.predict(input)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("headerfile", help="path and name of header file to be wirtten, example: ~/Weights.h", default = "./Weights.h")
    parser.add_argument("--test_size", type=int, help="the number of input data samples to generate in the header file", default= 250)
    parser.add_argument("--n_classes", type=int,  help="the number of classes contained in the training data", default=2)
    parser.add_argument("--n_features", type=int, help="the number of features in the input data", default = 3)
    parser.add_argument("--layer1", type=int, help ="the number of neurons in the 1st hidden layer", default = 10 )
    parser.add_argument("--layer2", type=int, help ="the number of neurons in the 2nd hidden layer", default = 5 )
    parser.add_argument("--n_epochs", type=int, help ="the number of training epochs", default = 10 )

    parser.add_argument("--dump_arrays", action="store_true", help="print tables to screen")


    args = parser.parse_args()

    trainX, testX, trainy, testy = generate_data(n_samples=int(args.test_size/0.25), n_classes=args.n_classes, n_features=args.n_features)

    model = generate_model(trainX, trainy, input_size=args.n_features, layer_1_neurons=args.layer1,
                           layer_2_neurons=args.layer2, layer_3_neurons=args.n_classes, n_epochs=args.n_epochs )



    write_header(testX, model, args.headerfile)

    if args.dump_arrays:
        dump_all(model, testX)

    evaluate_model(model, trainX, testX, trainy, testy)

if __name__ == "__main__":
    main()
