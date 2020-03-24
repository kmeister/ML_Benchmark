# ML_Benchmark
The goal of this project is to create a CPU benchmark for machine learning applications.

# Overview
the package contains a script called GenerateModel.py can generate an MLP with 3 hidden layers, and a few configurable hyperparameters. It'll train the MLP on a synthetic dataset, and then generate a header file containing a set of C arrays representing the weights of the trained model, and a test dataset

The benchmark folder contains a c++ class Layer, which can use the generated header file to run predictions on the test data

compile without the "DEBUG" declaration to create a version with no outputs to stderr or stdout for embedded applications

# To Build 
To build the benchmark, make sure you generate a Weights.h file in the Benchmark file first...
```
cd scripts
 ./ModelGenerator.py ../Benchmarks/Weights.h
 ```
use the following to see what options exist for configuring your model
```
./ModelGenerator.py -h 
```
